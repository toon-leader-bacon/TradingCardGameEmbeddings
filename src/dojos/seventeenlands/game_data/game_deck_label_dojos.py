"""Thin per-metric wrappers over GameDeckLabelMetric's three concrete
subclasses
(src/data_refinement/metrics/seventeenlands/game_data/game_deck_label_metrics.py).

Each wrapper only names METRIC, same thin-wrapper convention as
sts_gg/deck_label_dojos.py's: DeckWinPredictionDojo is a
DeckBinaryLabelMetricDojo, DeckGameLengthPredictionDojo is a
DeckLabelMetricDojo, and DeckRankTierPredictionDojo is a
DeckFixedLabelMetricDojo (all three in
../../generic/paired_metric_dojos.py). Each base reads its paired
metric's LABEL_COLUMN/DEFAULT_OUTPUT_PATH (and, for the fixed-
classification shape, LABEL_VALUES) ClassVars by reference, so none of
those are duplicated as literals here.

DeckLabelDataConstructor, NOT A NEW CONSTRUCTOR: GameDeckLabelMetric's
own row shape is (draft_id, match_number, game_number, deck_uuid,
LABEL_COLUMN) - extra id columns DeckLabelDataConstructor.build() never
reads (it only reads deck_uuid and the configured label column), so
sts_gg's DeckLabelDataConstructor (built for (run_id, deck_uuid,
LABEL_COLUMN)) is reused unchanged: shape-compatible, not
class-compatible.

CELL MAPPING: DeckWinPredictionMetric (bool) -> DeckBinaryLabelMetricDojo
(default label_caster=float handles True/False -> 1.0/0.0 fine, same as
sts_gg's WinDojo). DeckGameLengthPredictionMetric (int64) ->
DeckLabelMetricDojo. DeckRankTierPredictionMetric (str, closed
vocabulary + OTHER_LABEL) -> DeckFixedLabelMetricDojo, label_values
passed through unchanged - it already includes OTHER_LABEL (see that
metric's own docstring).
"""

from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckGameLengthPredictionMetric,
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.dojos.generic.paired_metric_dojos import (
    DeckBinaryLabelMetricDojo,
    DeckFixedLabelMetricDojo,
    DeckLabelMetricDojo,
)


class DeckGameLengthPredictionDojo(DeckLabelMetricDojo):
    """Deck -> predicted num_turns (DeckGameLengthPredictionMetric)."""

    METRIC = DeckGameLengthPredictionMetric


class DeckWinPredictionDojo(DeckBinaryLabelMetricDojo):
    """Deck -> predicted won (DeckWinPredictionMetric)."""

    METRIC = DeckWinPredictionMetric


class DeckRankTierPredictionDojo(DeckFixedLabelMetricDojo):
    """Deck -> predicted rank tier (DeckRankTierPredictionMetric).

    label_values=DeckRankTierPredictionMetric.LABEL_VALUES unchanged -
    it already includes OTHER_LABEL (see that class's own docstring)."""

    METRIC = DeckRankTierPredictionMetric
