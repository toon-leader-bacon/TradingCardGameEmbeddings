"""Thin dojo wrappers for the three GameDeckLabelMetric concrete
subclasses
(src/data_refinement/metrics/seventeenlands/game_data/game_deck_label_metrics.py).

Each wrapper only names METRIC, same thin-wrapper convention as
sts_gg/deck_label_dojos.py's, over the slice-aware deck bases in
../sliced_dojos.py: DeckWinPredictionDojo is a
SeventeenLandsDeckBinaryLabelDojo, DeckGameLengthPredictionDojo a
SeventeenLandsDeckRegressionDojo and DeckRankTierPredictionDojo a
SeventeenLandsDeckFixedLabelDojo. Each base reads its paired metric's
LABEL_COLUMN (and, for the fixed-classification shape, LABEL_VALUES)
ClassVars by reference and trains on the metric's slice file for its
data_slice (default all). Each row's deck_uuid is looked up in the
canonical MTG DeckBox (data/final/decks/mtg.db).

DeckLabelDataConstructor, NOT A NEW CONSTRUCTOR: GameDeckLabelMetric's
own row shape is (draft_id, match_number, game_number, deck_uuid,
<label>) - the extra draft_id/match_number/game_number columns are
simply ignored by a constructor that only reads deck_uuid and
LABEL_COLUMN, so sts_gg's DeckLabelDataConstructor (built as
DeckLabelDataConstructor(deck_box, LABEL_COLUMN)) is reused unchanged:
shape-compatible, not class-compatible.

CELL MAPPING: DeckWinPredictionMetric (bool) ->
SeventeenLandsDeckBinaryLabelDojo (default label_caster=float handles
True/False -> 1.0/0.0 fine, same as sts_gg's WinDojo).
DeckGameLengthPredictionMetric (int64) -> SeventeenLandsDeckRegressionDojo.
DeckRankTierPredictionMetric (str, closed vocabulary + OTHER_LABEL) ->
SeventeenLandsDeckFixedLabelDojo, label_values passed through unchanged -
it already includes OTHER_LABEL (see that metric's own docstring).
"""

from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckGameLengthPredictionMetric,
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.dojos.seventeenlands.sliced_dojos import (
    SeventeenLandsDeckBinaryLabelDojo,
    SeventeenLandsDeckFixedLabelDojo,
    SeventeenLandsDeckRegressionDojo,
)


class DeckGameLengthPredictionDojo(SeventeenLandsDeckRegressionDojo):
    """Deck -> predicted num_turns (DeckGameLengthPredictionMetric)."""

    METRIC = DeckGameLengthPredictionMetric


class DeckWinPredictionDojo(SeventeenLandsDeckBinaryLabelDojo):
    """Deck -> predicted won (DeckWinPredictionMetric)."""

    METRIC = DeckWinPredictionMetric


class DeckRankTierPredictionDojo(SeventeenLandsDeckFixedLabelDojo):
    """Deck -> predicted rank tier (DeckRankTierPredictionMetric).

    label_values=DeckRankTierPredictionMetric.LABEL_VALUES unchanged -
    it already includes OTHER_LABEL (see that class's own docstring)."""

    METRIC = DeckRankTierPredictionMetric
