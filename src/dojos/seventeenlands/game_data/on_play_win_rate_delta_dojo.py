"""Thin wrapper over OnPlayWinRateDeltaMetric
(src/data_refinement/metrics/seventeenlands/game_data/on_play_win_rate_delta_metric.py).

A CardAverageMetricDojo (../../generic/paired_metric_dojos.py) that
only names its paired metric, same thin-wrapper convention as
game_card_average_dojos.py's wrappers.

NULLABLE LABEL: a card never seen on one side of on_play writes None
for its delta (round-trips as NaN through the parquet float64 column) -
row_values.label_as_float() skips a NaN label the same
way it skips an unparseable one (see
src/dojos/generic/data_constructors/),
so those rows are silently dropped from training rather than corrupting
it with a NaN target.
"""

from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_delta_metric import (
    OnPlayWinRateDeltaMetric,
)
from src.dojos.generic.paired_metric_dojos import CardAverageMetricDojo


class OnPlayWinRateDeltaDojo(CardAverageMetricDojo):
    """Card -> predicted P(won | in deck, on_play) - P(won | in deck,
    on_draw) (OnPlayWinRateDeltaMetric)."""

    METRIC = OnPlayWinRateDeltaMetric
