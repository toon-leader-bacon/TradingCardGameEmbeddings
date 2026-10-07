"""Wrapper over GameLengthAssociationMetric
(src/data_refinement/metrics/seventeenlands/game_data/game_length_association_metric.py).

A SeventeenLandsCardLabelDojo (../sliced_dojos.py): it only names its
metric; the slice file supplies nocab_uuid, LABEL_COLUMN and
sample_count.
"""

from src.data_refinement.metrics.seventeenlands.game_data.game_length_association_metric import (
    GameLengthAssociationMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class GameLengthAssociationDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted (own average num_turns - the slice's average)
    (GameLengthAssociationMetric)."""

    METRIC = GameLengthAssociationMetric
