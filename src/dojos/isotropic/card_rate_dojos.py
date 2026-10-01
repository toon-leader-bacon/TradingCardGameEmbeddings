"""Per-metric wrappers over the isotropic single-card metrics: one card
in, one per-card number out (SingleCardRegressionDojo fed by
CardAverageDataConstructor).

Wrapped: AverageCopiesBoughtMetric, TurnCountAssociationMetric
(isotropic/summary/) and OpeningBuyRateMetric, PileExhaustionRateMetric
(isotropic/games/). Each row is (nocab_uuid, <label>, sample_count), the
shape CardAverageDataConstructor reads.

Not wrapped: VetoRateMetric (every row's veto_rate is 0.0 today, so
there is nothing to learn) and CopiesBoughtDistributionMetric (raw
per-deck samples of the quantity AverageCopiesBoughtMetric averages; an
MSE head on it would learn the same mean).

WHY NOT CardAverageMetricDojo: that base reads METRIC.LABEL_COLUMN, and
these metrics write their label column as a literal in their parquet
schema rather than as a LABEL_COLUMN ClassVar. Each wrapper here names
its metric's output path by reference and its label column once. If the
metrics gain LABEL_COLUMN ClassVars, every wrapper can become a
CardAverageMetricDojo with METRIC set and IsotropicCardRateDojo can go.
"""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.card_binder.card_binder import CardBinder
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
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec


class IsotropicCardRateDojo(SingleCardRegressionDojo):
    """Single card in -> LABEL_COLUMN (a per-card number) out.

    Subclasses set OUTPUT_PATH (read off the metric class) and
    LABEL_COLUMN.
    """

    OUTPUT_PATH: ClassVar[Path]
    LABEL_COLUMN: ClassVar[str]

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs:
            card_binder: card lookup for the Dominion nocab_uuids.
            holdout: card holdout shared by every dojo in a run.
            card_embedding_size: width of the encoder's card embeddings.
            path_to_training_data: overrides OUTPUT_PATH.
            name: Trainer-facing name and split-file prefix; None falls
                back to the training file's stem (see DojoConfig.name).
            rng_seed: split/shuffle seed; None means unseeded.
            strict_version_check: raise (rather than warn) on a metric
                file built against a different CardBinder version.
        Output: none (constructor).
        Side effects: see SingleCardRegressionDojo (may write split files).
        Exceptions: see SingleCardRegressionDojo.

        Example:
            >>> OpeningBuyRateDojo(binder, HoldoutSpec.no_holdout(), 32)
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data or self.OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(self.LABEL_COLUMN),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class AverageCopiesBoughtDojo(IsotropicCardRateDojo):
    """Card -> average copies in a final deck that has it
    (AverageCopiesBoughtMetric). Covers basic cards too (172 rows)."""

    OUTPUT_PATH = AverageCopiesBoughtMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "average_copies_bought"


class TurnCountAssociationDojo(IsotropicCardRateDojo):
    """Card -> signed shift in the winner's turn count when the card is in
    the kingdom, against the corpus average (TurnCountAssociationMetric).
    Positive means slower games."""

    OUTPUT_PATH = TurnCountAssociationMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "turn_count_delta"


class OpeningBuyRateDojo(IsotropicCardRateDojo):
    """Card -> P(opened with | in kingdom) (OpeningBuyRateMetric).

    A regression cell despite the label being a rate in [0, 1], same
    choice as sts_gg's CardWinRateDojo."""

    OUTPUT_PATH = OpeningBuyRateMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "opening_buy_rate"


class PileExhaustionRateDojo(IsotropicCardRateDojo):
    """Card -> P(its pile is empty at game end | in kingdom)
    (PileExhaustionRateMetric). A rate in [0, 1], regressed."""

    OUTPUT_PATH = PileExhaustionRateMetric.DEFAULT_OUTPUT_PATH
    LABEL_COLUMN = "pile_exhaustion_rate"
