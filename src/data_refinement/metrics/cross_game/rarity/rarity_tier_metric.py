"""RarityTierMetric: every translated game's cards, labeled with their
RarityTier. The "medium" label set for intrinsic evaluation and the
source file of the cross-game rarity dojo.

A CorpusScanMetric (../../generic/corpus_scan_metric.py): the whole
corpus is an already-loaded card store, so one scan() call writes the
file. Unlike MaskedFieldMetric it spans games, so its output is one
parquet for all of them, stamped with one binder version per game
(../../version_metadata.py, MultiGameVersionMetadata).

Output columns, one row per card that has a rarity value:

    nocab_uuid   string   the card
    source_game  string   GameId value; the dojo balances and masks by it
    raw_rarity   string   the binder's value, for inspection
    label        string   RarityTier value (OTHER rows are written; the
                          dojo skips them)

A game with no translator (Dominion) and a card with no rarity value
(including the Unknown sentinel) get no row.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import ClassVar, Mapping

import pandas as pd
from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.metrics.cross_game.rarity.rarity_translator import (
    RarityTranslator,
)
from src.data_refinement.metrics.cross_game.rarity.translator_tables import (
    RARITY_TRANSLATORS,
)
from src.data_refinement.metrics.version_metadata import (
    MultiGameVersionMetadata,
    write_dataframe_with_multi_game_versions,
)
from src.schema.game_id import GameId
from src.schema.rarity_tier import RarityTier


@dataclass(frozen=True)
class _RarityTierRow:
    """One output row: a card, its game, its raw rarity, and its tier.

    Inputs: none (data holder). Output: n/a. Side effects: none.
    Exceptions: none.
    """

    nocab_uuid: str
    source_game: str
    raw_rarity: str
    label: str


class RarityTierMetric:
    """Card -> RarityTier, over every game that has a RarityTranslator.

    Satisfies the CorpusScanMetric Protocol structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/cross_game/rarity_tier.parquet"
    )
    LABEL_COLUMN: ClassVar[str] = "label"
    LABEL_VALUES: ClassVar[tuple[str, ...]] = tuple(tier.value for tier in RarityTier)

    def __init__(
        self,
        card_lookup: CardLookup,
        translators: Mapping[GameId, RarityTranslator] = RARITY_TRANSLATORS,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_lookup: a card store already holding every translated
                game's cards (one CardBinder loaded from each game's
                file satisfies this).
            translators: per game, how to read its rarity. Defaults to
                the six games in translator_tables.py.
            output_path: overrides DEFAULT_OUTPUT_PATH.
        Output: none (constructor).
        Side effects: none - no I/O until scan().
        Exceptions: none.
        """
        self._card_lookup = card_lookup
        self._translators = translators
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH

    def scan(self) -> Path:
        """Label every translated game's cards and write the file.

        Inputs: none (uses the constructor's lookup and translators).
        Output: the path written.
        Side effects: creates the output's parent directories; writes one
            parquet (columns above) stamped with each translated game's
            CardLookup.version_for(); shows a tqdm bar per game.
        Exceptions: UnmappedRarityError for a raw rarity a game's table
            does not list - raised before anything is written, so a
            failed run leaves the previous file untouched;
            whatever pyarrow raises writing the file.

        Example:
            >>> RarityTierMetric(binder).scan()
            PosixPath('data/metrics/cross_game/rarity_tier.parquet')
        """
        rows: list[_RarityTierRow] = []

        # Each game in GameId order, so the file is reproducible
        for game in sorted(self._translators, key=lambda game: game.value):
            rows.extend(self._rows_for_game(game, self._translators[game]))

        # An empty frame still needs the columns, so the file's schema holds
        frame = pd.DataFrame(
            [asdict(row) for row in rows],
            columns=[field.name for field in fields(_RarityTierRow)],
        )
        write_dataframe_with_multi_game_versions(
            frame, self._output_path, self._binder_versions()
        )
        return self._output_path

    def _rows_for_game(
        self, game: GameId, translator: RarityTranslator
    ) -> list[_RarityTierRow]:
        """One game's rows: every card with a rarity value, labeled.

        Inputs: game, its translator.
        Output: rows in the lookup's card order; source_game and label are
            written as their enum .value strings (asdict does not do it).
        Side effects: none (a tqdm bar to stderr).
        Exceptions: UnmappedRarityError from the translator.
        """
        result: list[_RarityTierRow] = []

        # The Unknown sentinel has no raw_content, so no rarity
        cards = list(self._card_lookup.all_cards(game))
        for card in tqdm(cards, desc=f"rarity_tier {game.value}", unit="card"):
            tier = translator.rarity_tier_of(card)
            if tier is None:
                continue
            raw_rarity = translator.raw_rarity_of(card)
            assert raw_rarity is not None  # a tier implies a raw value
            result.append(
                _RarityTierRow(
                    nocab_uuid=str(card.nocab_uuid),
                    source_game=game.value,
                    raw_rarity=raw_rarity,
                    label=tier.value,
                )
            )
        return result

    def _binder_versions(self) -> MultiGameVersionMetadata:
        """CardLookup.version_for() of every translated game.

        Inputs: none. Output: MultiGameVersionMetadata.
        Side effects: none. Exceptions: none.
        """
        return MultiGameVersionMetadata(
            {game: self._card_lookup.version_for(game) for game in self._translators}
        )
