"""Wrapper over TutorTargetRateMetric
(src/data_refinement/metrics/seventeenlands/game_data/tutor_target_rate_metric.py).

Distinct class from
src/dojos/seventeenlands/replay_data/tutor_target_rate_dojo.py's own
TutorTargetRateDojo - each lives in its own source-specific package.

A SeventeenLandsCardLabelDojo (../sliced_dojos.py): it only names its
metric; the slice file supplies nocab_uuid, LABEL_COLUMN and
sample_count.
"""

from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class TutorTargetRateDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted P(tutored | in deck) (TutorTargetRateMetric)."""

    METRIC = TutorTargetRateMetric
