"""Thin wrapper over TutorTargetRateMetric
(src/data_refinement/metrics/seventeenlands/game_data/tutor_target_rate_metric.py).

A CardAverageMetricDojo (../../generic/paired_metric_dojos.py) that
only names its paired metric, same thin-wrapper convention as
game_card_average_dojos.py's wrappers.

Distinct class from
src/dojos/seventeenlands/replay_data/tutor_target_rate_dojo.py's own
TutorTargetRateDojo - each lives in its own source-specific package,
mirroring the two paired metric classes' own naming convention (see
that metric's module docstring).
"""

from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.dojos.generic.paired_metric_dojos import CardAverageMetricDojo


class TutorTargetRateDojo(CardAverageMetricDojo):
    """Card -> predicted P(tutored | in deck) (TutorTargetRateMetric)."""

    METRIC = TutorTargetRateMetric
