"""Per-metric wrappers over ReplayTurnEventRateMetric's two concrete
subclasses
(src/data_refinement/metrics/seventeenlands/replay_data/replay_turn_event_rate_metrics.py).

Each is a CardAverageMetricDojo (../../generic/paired_metric_dojos.py)
that only names its paired metric.
"""

from src.data_refinement.metrics.seventeenlands.replay_data.replay_turn_event_rate_metrics import (
    CombatDamagePushThroughRateMetric,
    CombatKillInvolvementRateMetric,
)
from src.dojos.generic.paired_metric_dojos import CardAverageMetricDojo


class CombatKillInvolvementRateDojo(CardAverageMetricDojo):
    """Card -> predicted P(a creature died in combat that half-turn |
    card fought that half-turn) (CombatKillInvolvementRateMetric)."""

    METRIC = CombatKillInvolvementRateMetric


class CombatDamagePushThroughRateDojo(CardAverageMetricDojo):
    """Card -> predicted P(card in creatures_unblocked | card in
    creatures_attacked) (CombatDamagePushThroughRateMetric)."""

    METRIC = CombatDamagePushThroughRateMetric
