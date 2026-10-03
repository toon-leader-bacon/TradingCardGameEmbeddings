"""Wrapper over OnPlayWinRateSensitivityByDeckMetric
(src/data_refinement/metrics/seventeenlands/game_data/on_play_win_rate_sensitivity_by_deck_metric.py).

A SeventeenLandsDeckRegressionDojo (../sliced_dojos.py): it only names
its metric.

NULLABLE LABEL: a deck never seen on one side of on_play has a null
sensitivity (NaN in the float64 column); DeckLabelDataConstructor.build()
skips those rows (see src/dojos/generic/data_constructors/).
"""

from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_metric import (  # noqa: E501
    OnPlayWinRateSensitivityByDeckMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsDeckRegressionDojo


class OnPlayWinRateSensitivityByDeckDojo(SeventeenLandsDeckRegressionDojo):
    """Deck -> predicted P(won | on_play) - P(won | on_draw)
    (OnPlayWinRateSensitivityByDeckMetric)."""

    METRIC = OnPlayWinRateSensitivityByDeckMetric
