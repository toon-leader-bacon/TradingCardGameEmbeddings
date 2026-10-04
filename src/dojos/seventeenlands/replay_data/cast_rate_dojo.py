"""Wrapper over CastRateMetric
(src/data_refinement/metrics/seventeenlands/replay_data/cast_rate_metric.py).

A SeventeenLandsCardLabelDojo (../sliced_dojos.py): it only names its
metric; the slice file supplies nocab_uuid, LABEL_COLUMN and
sample_count.
"""

from src.data_refinement.metrics.seventeenlands.replay_data.cast_rate_metric import (
    CastRateMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class CastRateDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted P(the user casts it | in deck) (CastRateMetric)."""

    METRIC = CastRateMetric
