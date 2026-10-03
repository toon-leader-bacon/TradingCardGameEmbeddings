"""Thin per-metric wrappers over GameDeckLabelMetric's three concrete
subclasses
(src/data_refinement/metrics/seventeenlands/game_data/game_deck_label_metrics.py).

Each wrapper subclasses whichever generic multi-card cell fits its
paired metric's label type and adds only configuration: the metric's
LABEL_COLUMN (and, for DeckRankTierPredictionMetric, LABEL_VALUES)
ClassVars by reference, a DeckLabelDataConstructor for that one label
column, and a data_slice (default all) whose slice file it trains on.
The regression one is a SeventeenLandsDeckRegressionDojo
(../sliced_dojos.py); the two classification ones call
seventeenlands_training_path() themselves.

DeckLabelDataConstructor, NOT A NEW CONSTRUCTOR: GameDeckLabelMetric's
own row shape is (draft_id, match_number, game_number, deck_uuid,
LABEL_COLUMN) - extra id columns DeckLabelDataConstructor.build() never
reads (it only reads deck_uuid and the configured label column), so
sts_gg's DeckLabelDataConstructor (built for (run_id, deck_uuid,
LABEL_COLUMN)) is reused unchanged: shape-compatible, not
class-compatible.

CELL MAPPING: DeckWinPredictionMetric (bool) -> MultiCardBinaryClassificationDojo
(default label_caster=float handles True/False -> 1.0/0.0 fine, same as
sts_gg's WinDojo). DeckGameLengthPredictionMetric (int64) ->
SeventeenLandsDeckRegressionDojo. DeckRankTierPredictionMetric (str, closed
vocabulary + OTHER_LABEL) -> MultiCardFixedClassificationDojo
(label_caster=str), label_values passed through unchanged - it already
includes OTHER_LABEL (see that metric's own docstring).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.data_slice import (
    SeventeenLandsSlice,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckGameLengthPredictionMetric,
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.seventeenlands.sliced_dojos import (
    ALL_DATA,
    SeventeenLandsDeckRegressionDojo,
    seventeenlands_training_path,
)
from src.schema.holdout import HoldoutSpec


class DeckGameLengthPredictionDojo(SeventeenLandsDeckRegressionDojo):
    """Deck -> predicted num_turns (DeckGameLengthPredictionMetric)."""

    METRIC = DeckGameLengthPredictionMetric


class DeckWinPredictionDojo(MultiCardBinaryClassificationDojo):
    """Deck -> predicted won (DeckWinPredictionMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        data_slice: SeventeenLandsSlice = ALL_DATA,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs: as SeventeenLandsDeckRegressionDojo
            (../sliced_dojos.py): card_binder, holdout, deck_box (the
            family DeckBox), card_embedding_size, data_slice (default
            all), path_to_training_data (overrides data_slice), name,
            rng_seed, strict_version_check.
        Output: none (constructor).
        Side effects: may build the slice file; see MultiCardBinaryClassificationDojo
            (may write split files).
        Exceptions: as seventeenlands_training_path and MultiCardBinaryClassificationDojo.

        Example:
            >>> DeckWinPredictionDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=seventeenlands_training_path(
                DeckWinPredictionMetric, data_slice, path_to_training_data
            ),
            data_constructor=DeckLabelDataConstructor(
                deck_box, DeckWinPredictionMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class DeckRankTierPredictionDojo(MultiCardFixedClassificationDojo):
    """Deck -> predicted rank tier (DeckRankTierPredictionMetric).

    label_values=DeckRankTierPredictionMetric.LABEL_VALUES unchanged -
    it already includes OTHER_LABEL (see that class's own docstring)."""

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        data_slice: SeventeenLandsSlice = ALL_DATA,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs: as SeventeenLandsDeckRegressionDojo
            (../sliced_dojos.py): card_binder, holdout, deck_box (the
            family DeckBox), card_embedding_size, data_slice (default
            all), path_to_training_data (overrides data_slice), name,
            rng_seed, strict_version_check.
        Output: none (constructor).
        Side effects: may build the slice file; see MultiCardFixedClassificationDojo
            (may write split files).
        Exceptions: as seventeenlands_training_path and MultiCardFixedClassificationDojo.

        Example:
            >>> DeckRankTierPredictionDojo(binder, HoldoutSpec.no_holdout(), box, 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=seventeenlands_training_path(
                DeckRankTierPredictionMetric, data_slice, path_to_training_data
            ),
            data_constructor=DeckLabelDataConstructor(
                deck_box,
                DeckRankTierPredictionMetric.LABEL_COLUMN,
                label_caster=str,
            ),
            label_values=DeckRankTierPredictionMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
