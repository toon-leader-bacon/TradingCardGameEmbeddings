"""Per-metric wrappers over GameCardAverageMetric's three plain
win-rate-shaped concrete subclasses
(src/data_refinement/metrics/seventeenlands/game_data/game_card_average_metrics.py).

Each is a CardAverageMetricDojo (../../generic/paired_metric_dojos.py)
that only names its paired metric. GameLengthAssociationMetric, this
family's fourth subclass, is wrapped in game_length_association_dojo.py,
mirroring that metric's own separate module.
"""

from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metrics import (
    DrawnWinRateMetric,
    OpeningHandWinRateMetric,
    WinRateWhenInDeckMetric,
)
from src.dojos.generic.paired_metric_dojos import CardAverageMetricDojo


class WinRateWhenInDeckDojo(CardAverageMetricDojo):
    """Card -> predicted P(won | card in deck_<name>) (WinRateWhenInDeckMetric)."""

    METRIC = WinRateWhenInDeckMetric


class OpeningHandWinRateDojo(CardAverageMetricDojo):
    """Card -> predicted P(won | card in opening_hand_<name>) (OpeningHandWinRateMetric)."""

    METRIC = OpeningHandWinRateMetric


class DrawnWinRateDojo(CardAverageMetricDojo):
    """Card -> predicted P(won | card in drawn_<name>) (DrawnWinRateMetric)."""

    METRIC = DrawnWinRateMetric
