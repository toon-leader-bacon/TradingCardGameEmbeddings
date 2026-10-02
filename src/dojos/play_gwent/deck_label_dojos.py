"""Thin wrapper over GuideVotesMetric
(src/data_refinement/metrics/play_gwent/guide_votes_metric.py). Its rows
point into the published Gwent deck box, which the catalog passes in
(read-only).
"""

from src.data_refinement.metrics.play_gwent.guide_votes_metric import (
    GuideVotesMetric,
)
from src.dojos.generic.paired_metric_dojos import DeckLabelMetricDojo


class GuideVotesDojo(DeckLabelMetricDojo):
    """Guide deck -> sign(votes) * ln(1 + |votes|) (deck reception)."""

    METRIC = GuideVotesMetric
