"""Wrapper over DiscardRateMetric
(src/data_refinement/metrics/seventeenlands/replay_data/discard_rate_metric.py).

A SeventeenLandsCardLabelDojo (../sliced_dojos.py): it only names its
metric; the slice file supplies nocab_uuid, LABEL_COLUMN and
sample_count.
"""

from src.data_refinement.metrics.seventeenlands.replay_data.discard_rate_metric import (
    DiscardRateMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class DiscardRateDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted P(the user discards it | in deck) (DiscardRateMetric)."""

    METRIC = DiscardRateMetric
