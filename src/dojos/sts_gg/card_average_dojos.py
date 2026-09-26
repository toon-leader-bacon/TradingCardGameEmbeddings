"""Per-metric wrappers over CardAverageMetric's nine concrete subclasses
(src/data_refinement/metrics/sts_gg/card_average_metrics.py).

Each is a CardAverageMetricDojo (../generic/paired_metric_dojos.py)
that only names its paired metric.
"""

from src.data_refinement.metrics.sts_gg.card_average_metrics import (
    CardDeckSizeMetric,
    CardElitesKilledMetric,
    CardFloorsClearedMetric,
    CardRelicCountMetric,
    CardTotalCardsPickedMetric,
    CardTotalCombatsMetric,
    CardTotalDamageTakenMetric,
    CardTotalTurnsMetric,
    CardWinRateMetric,
)
from src.dojos.generic.paired_metric_dojos import CardAverageMetricDojo


class CardRelicCountDojo(CardAverageMetricDojo):
    """Card -> predicted average relicCount (CardRelicCountMetric)."""

    METRIC = CardRelicCountMetric


class CardTotalDamageTakenDojo(CardAverageMetricDojo):
    """Card -> predicted average stats.totalDamageTaken (CardTotalDamageTakenMetric)."""

    METRIC = CardTotalDamageTakenMetric


class CardDeckSizeDojo(CardAverageMetricDojo):
    """Card -> predicted average deckSize (CardDeckSizeMetric)."""

    METRIC = CardDeckSizeMetric


class CardTotalCardsPickedDojo(CardAverageMetricDojo):
    """Card -> predicted average stats.totalCardsPicked (CardTotalCardsPickedMetric)."""

    METRIC = CardTotalCardsPickedMetric


class CardTotalTurnsDojo(CardAverageMetricDojo):
    """Card -> predicted average stats.totalTurns (CardTotalTurnsMetric)."""

    METRIC = CardTotalTurnsMetric


class CardElitesKilledDojo(CardAverageMetricDojo):
    """Card -> predicted average stats.elitesKilled (CardElitesKilledMetric)."""

    METRIC = CardElitesKilledMetric


class CardFloorsClearedDojo(CardAverageMetricDojo):
    """Card -> predicted average stats.floorsCleared (CardFloorsClearedMetric)."""

    METRIC = CardFloorsClearedMetric


class CardTotalCombatsDojo(CardAverageMetricDojo):
    """Card -> predicted average stats.totalCombats (CardTotalCombatsMetric)."""

    METRIC = CardTotalCombatsMetric


class CardWinRateDojo(CardAverageMetricDojo):
    """Card -> predicted P(win | card in final deck) (CardWinRateMetric).

    A regression cell despite the label being a rate in [0, 1] - see
    CardAverageMetric's WIN RATE IS AN AVERAGE docstring note for why a
    win/loss indicator is folded through the same averaging machinery
    as every other CardAverageMetric subclass, rather than treated as a
    two-class classification problem.
    """

    METRIC = CardWinRateMetric
