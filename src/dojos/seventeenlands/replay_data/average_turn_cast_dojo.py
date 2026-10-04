"""Wrapper over AverageTurnCastMetric
(src/data_refinement/metrics/seventeenlands/replay_data/average_turn_cast_metric.py).

A SeventeenLandsCardLabelDojo (../sliced_dojos.py): it only names its
metric; the slice file supplies nocab_uuid, LABEL_COLUMN and
sample_count.
"""

from src.data_refinement.metrics.seventeenlands.replay_data.average_turn_cast_metric import (
    AverageTurnCastMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class AverageTurnCastDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted average turn it is cast on (AverageTurnCastMetric)."""

    METRIC = AverageTurnCastMetric
