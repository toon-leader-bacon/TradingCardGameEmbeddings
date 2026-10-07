"""Wrapper over TutorTargetRateMetric
(src/data_refinement/metrics/seventeenlands/replay_data/tutor_target_rate_metric.py).

A SeventeenLandsCardLabelDojo (../sliced_dojos.py): it only names its
metric; the slice file supplies nocab_uuid, LABEL_COLUMN and
sample_count.
"""

from src.data_refinement.metrics.seventeenlands.replay_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class TutorTargetRateDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted P(the user tutors it | in deck)
    (replay_data's TutorTargetRateMetric)."""

    METRIC = TutorTargetRateMetric
