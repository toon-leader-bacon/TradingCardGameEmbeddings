"""Thin per-metric wrappers over GameDeckLabelMetric's three concrete
subclasses
(src/data_refinement/metrics/seventeenlands/game_data/game_deck_label_metrics.py).

Each wrapper subclasses whichever generic multi-card cell fits its
paired metric's label type and adds only configuration: the metric's
LABEL_COLUMN/DEFAULT_OUTPUT_PATH (and, for DeckRankTierPredictionMetric,
LABEL_VALUES) ClassVars by reference, and a DeckLabelDataConstructor for
that one label column. The regression one is a DeckLabelMetricDojo
(../../generic/paired_metric_dojos.py).

DeckLabelDataConstructor, NOT A NEW CONSTRUCTOR: GameDeckLabelMetric's
own row shape is (draft_id, match_number, game_number, deck_uuid,
LABEL_COLUMN) - extra id columns DeckLabelDataConstructor.build() never
reads (it only reads deck_uuid and the configured label column), so
sts_gg's DeckLabelDataConstructor (built for (run_id, deck_uuid,
LABEL_COLUMN)) is reused unchanged: shape-compatible, not
class-compatible (see plans/seventeen_lands_dojos.md).

CELL MAPPING: DeckWinPredictionMetric (bool) -> MultiCardBinaryClassificationDojo
(default label_caster=float handles True/False -> 1.0/0.0 fine, same as
sts_gg's WinDojo). DeckGameLengthPredictionMetric (int64) ->
DeckLabelMetricDojo. DeckRankTierPredictionMetric (str, closed
vocabulary + OTHER_LABEL) -> MultiCardFixedClassificationDojo
(label_caster=str), label_values passed through unchanged - it already
includes OTHER_LABEL (see that metric's own docstring).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckGameLengthPredictionMetric,
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.dojos.generic.paired_metric_dojos import DeckLabelMetricDojo
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.schema.holdout import HoldoutSpec


class DeckGameLengthPredictionDojo(DeckLabelMetricDojo):
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
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or DeckWinPredictionMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                deck_box, DeckWinPredictionMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
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
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or DeckRankTierPredictionMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                deck_box,
                DeckRankTierPredictionMetric.LABEL_COLUMN,
                label_caster=str,
            ),
            label_values=DeckRankTierPredictionMetric.LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
