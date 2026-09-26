"""Wrapper over GameLengthAssociationMetric
(src/data_refinement/metrics/seventeenlands/game_data/game_length_association_metric.py).

A CardAverageMetricDojo (../../generic/paired_metric_dojos.py): this
metric shares the card-average output row shape
(nocab_uuid/LABEL_COLUMN/sample_count) even though it overrides
finalize() to subtract a format-wide baseline.
"""

from src.data_refinement.metrics.seventeenlands.game_data.game_length_association_metric import (
    GameLengthAssociationMetric,
)
from src.dojos.generic.paired_metric_dojos import CardAverageMetricDojo


class GameLengthAssociationDojo(CardAverageMetricDojo):
    """Card -> predicted (own average num_turns - format-wide average)
    (GameLengthAssociationMetric)."""

    METRIC = GameLengthAssociationMetric
