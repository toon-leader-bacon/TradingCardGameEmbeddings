"""Per-metric wrappers over the isotropic single-card metrics: one card
in, one per-card number out (CardAverageMetricDojo).

Wrapped: AverageCopiesBoughtMetric, TurnCountAssociationMetric
(isotropic/summary/) and OpeningBuyRateMetric, PileExhaustionRateMetric
(isotropic/games/), VetoRateMetric (isotropic/summary/).

Not wrapped: CopiesBoughtDistributionMetric (raw
per-deck samples of the quantity AverageCopiesBoughtMetric averages; an
MSE head on it would learn the same mean).
"""

from src.data_refinement.metrics.isotropic.games.opening_buy_rate_metric import (
    OpeningBuyRateMetric,
)
from src.data_refinement.metrics.isotropic.games.pile_exhaustion_rate_metric import (
    PileExhaustionRateMetric,
)
from src.data_refinement.metrics.isotropic.summary.average_copies_bought_metric import (
    AverageCopiesBoughtMetric,
)
from src.data_refinement.metrics.isotropic.summary.turn_count_association_metric import (
    TurnCountAssociationMetric,
)
from src.data_refinement.metrics.isotropic.summary.veto_rate_metric import (
    VetoRateMetric,
)
from src.dojos.generic.paired_metric_dojos import CardAverageMetricDojo


class AverageCopiesBoughtDojo(CardAverageMetricDojo):
    """Card -> average copies in a final deck that has it
    (AverageCopiesBoughtMetric). Covers basic cards too (172 rows)."""

    METRIC = AverageCopiesBoughtMetric


class TurnCountAssociationDojo(CardAverageMetricDojo):
    """Card -> signed shift in the winner's turn count when the card is in
    the kingdom, against the corpus average (TurnCountAssociationMetric).
    Positive means slower games."""

    METRIC = TurnCountAssociationMetric


class OpeningBuyRateDojo(CardAverageMetricDojo):
    """Card -> P(opened with | in kingdom) (OpeningBuyRateMetric).

    A regression cell despite the label being a rate in [0, 1], same
    choice as sts_gg's CardWinRateDojo."""

    METRIC = OpeningBuyRateMetric


class PileExhaustionRateDojo(CardAverageMetricDojo):
    """Card -> P(its pile is empty at game end | in kingdom)
    (PileExhaustionRateMetric). A rate in [0, 1], regressed."""

    METRIC = PileExhaustionRateMetric


class VetoRateDojo(CardAverageMetricDojo):
    """Card -> P(vetoed | offered) in games with vetoing on
    (VetoRateMetric). A rate in [0, 1], regressed."""

    METRIC = VetoRateMetric
