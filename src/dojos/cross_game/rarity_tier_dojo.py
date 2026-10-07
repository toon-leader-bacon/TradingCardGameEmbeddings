"""Thin wrapper over RarityTierMetric
(src/data_refinement/metrics/cross_game/rarity/rarity_tier_metric.py).

One SingleCardFixedClassificationDojo with one head over the five
trained classes (TIER_1..TIER_4 and SPECIAL), fed cards from every game
the metric covers. Three things differ from the per-game masked-field
wrappers:

- The card lookup spans games (MultiGameCardLookup), and the metric file
  carries one binder version per game.
- What is masked differs per game, so the mask is a PerGameMod over each
  translator's masked_paths(), applied on every split (train_only=False)
  so no split can read the answer off raw_content.
- TRAIN rows are drawn evenly across games (GameBalancedChunkReader), for
  training and for the loss calibration, so MTG's 35k cards do not drown
  StS2's ~600. TEST and VALIDATION are read in file order.
"""

import random
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.metrics.cross_game.rarity.rarity_tier_metric import (
    RarityTierMetric,
)
from src.data_refinement.metrics.cross_game.rarity.rarity_translator import (
    RarityTranslator,
)
from src.data_refinement.metrics.cross_game.rarity.translator_tables import (
    RARITY_TRANSLATORS,
)
from src.dojos.cross_game.rarity_tier_data_constructor import (
    RarityTierDataConstructor,
    is_trainable_row,
)
from src.dojos.file_managers.game_balanced_chunk_reader import GameBalancedChunkReader
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.generic_dojo import ROWS_PER_CHUNK
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.dojos.mods.per_game_mod import PerGameMod
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.rarity_tier import LADDER, RarityTier
from src.schema.splits import Split

# The metric's column the TRAIN draw balances over
_BALANCE_COLUMN = "source_game"


class RarityTierDojo(SingleCardFixedClassificationDojo):
    """Card, its rarity masked -> its RarityTier (RarityTierMetric).

    label_values = TIER_1..TIER_4 then SPECIAL. OTHER rows are never
    trained on (RarityTierDataConstructor).

    The loss baseline is fit to the game-balanced TRAIN draw, while TEST and
    VALIDATION are read in file order (MTG-dominated), so loss / baseline is
    not strictly like-for-like across splits.

    Inputs (constructor): card_lookup (a CardLookup spanning every game in
        translators; MultiGameCardLookup from the catalog), holdout,
        card_embedding_size, path_to_training_data (overrides the metric's
        DEFAULT_OUTPUT_PATH), name, rng_seed, strict_version_check,
        translators (game -> RarityTranslator; defaults to the six games).
    Output: n/a.
    Side effects: may write split files; reads a strided TRAIN sample to
        calibrate the loss (see GenericDojo).
    Exceptions: see SingleCardFixedClassificationDojo; ValueError from
        GenericDojo._check_multi_game_versions if the metric file's
        recorded binder version for any game is stale.

    Example:
        >>> RarityTierDojo(lookup, HoldoutSpec.no_holdout(), 32)
    """

    def __init__(
        self,
        card_lookup: CardLookup,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
        translators: Mapping[GameId, RarityTranslator] = RARITY_TRANSLATORS,
    ) -> None:
        # Set before super().__init__: its loss calibration already reads
        # TRAIN through _chunks
        self._balance_rng = random.Random(rng_seed)

        super().__init__(
            card_lookup=card_lookup,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or RarityTierMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=RarityTierDataConstructor(),
            label_values=[tier.value for tier in (*LADDER, RarityTier.SPECIAL)],
            card_embedding_size=card_embedding_size,
            mod_pipeline=ModPipeline(
                [PerGameMod(_masking_pipelines(translators), train_only=False)]
            ),
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )

    def _chunks(self, split: Split) -> Iterable[pd.DataFrame]:
        """TRAIN: rows drawn evenly across games; other splits: file order.

        Inputs: split. Output: DataFrame chunks; TRAIN's sum to its
            non-OTHER row count, with each game equally likely per draw.
        Side effects: reads the split file; advances the balance rng for
            TRAIN.
        Exceptions: FileNotFoundError if the split files are missing.
        """
        if split is not Split.TRAIN:
            return super()._chunks(split)
        # OTHER rows are removed before balancing, so every game's share of
        # the trained examples is even
        return GameBalancedChunkReader(
            self.file_manager.split_path(split),
            _BALANCE_COLUMN,
            ROWS_PER_CHUNK,
            self._balance_rng,
            keep_row=is_trainable_row,
        )


def _masking_pipelines(
    translators: Mapping[GameId, RarityTranslator],
) -> dict[GameId, ModPipeline]:
    """Per game, the mods that hide that game's rarity from a card.

    Inputs: translators (game -> its RarityTranslator).
    Output: game -> ModPipeline of one MaskTargetKeyMod per masked path of
        that game's translator (train_only=False; a missing path is added
        as the mask token, so every card of a game looks alike).
    Side effects: none. Exceptions: none.

    Example:
        >>> _masking_pipelines(RARITY_TRANSLATORS)[GameId.GWENT].mods[0].key
        ('rarity',)
    """
    return {
        game: ModPipeline(
            [
                MaskTargetKeyMod(key=path, train_only=False)
                for path in translator.masked_paths()
            ]
        )
        for game, translator in translators.items()
    }
