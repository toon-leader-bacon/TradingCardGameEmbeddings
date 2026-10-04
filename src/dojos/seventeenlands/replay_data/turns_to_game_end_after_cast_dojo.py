"""Wrapper over TurnsToGameEndAfterCastMetric
(src/data_refinement/metrics/seventeenlands/replay_data/turns_to_game_end_after_cast_metric.py).

A SeventeenLandsCardLabelDojo (../sliced_dojos.py): it only names its
metric; the slice file supplies nocab_uuid, LABEL_COLUMN and
sample_count.
"""

from src.data_refinement.metrics.seventeenlands.replay_data.turns_to_game_end_after_cast_metric import (  # noqa: E501
    TurnsToGameEndAfterCastMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class TurnsToGameEndAfterCastDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted turns left after its first cast
    (TurnsToGameEndAfterCastMetric)."""

    METRIC = TurnsToGameEndAfterCastMetric
