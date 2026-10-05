"""Thin wrapper over OnPlayWinRateSensitivityByDeckMetric
(src/data_refinement/metrics/seventeenlands/game_data/on_play_win_rate_sensitivity_by_deck_metric.py).

A DeckLabelMetricDojo (../../generic/paired_metric_dojos.py) that only
names its paired metric, same thin-wrapper convention as
sts_gg/deck_label_dojos.py's wrappers.

NULLABLE LABEL: a deck never seen on one side of on_play writes None
for its sensitivity (round-trips as NaN through the parquet float64
column) - DeckLabelDataConstructor.build() now skips a NaN label the
same way it skips an unresolvable deck_uuid (see
src/dojos/generic/data_constructors/).
"""

from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_metric import (  # noqa: E501
    OnPlayWinRateSensitivityByDeckMetric,
)
from src.dojos.generic.paired_metric_dojos import DeckLabelMetricDojo


class OnPlayWinRateSensitivityByDeckDojo(DeckLabelMetricDojo):
    """Deck -> predicted P(won | on_play) - P(won | on_draw)
    (OnPlayWinRateSensitivityByDeckMetric)."""

    METRIC = OnPlayWinRateSensitivityByDeckMetric
