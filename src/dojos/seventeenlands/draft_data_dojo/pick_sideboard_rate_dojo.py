"""Thin MetricRegressionDojo wrapper for PickSideboardRateMetric.

See plans/pipeline_conventions.md and average_pick_number_dojo.py's
own docstring for the rationale this mirrors exactly — this class
exists purely for discoverability, not new logic.
"""

from pathlib import Path

import torch

from src.data_refinement.seventeenlands.draft_data_metrics.draft_metric_scanner import (
    DraftMetricScanner,
)
from src.data_refinement.seventeenlands.draft_data_metrics.metrics.pick_sideboard_rate import (
    PickSideboardRateMetric,
)
from src.dojos.losses.loss import Loss
from src.dojos.losses.mse_loss import MSELoss
from src.dojos.seventeenlands.metric_regression_dojo import MetricRegressionDojo


class PickSideboardRateDojo(MetricRegressionDojo):
    """Regresses a card's embedding against its pick_sideboard_rate.

    Not a new capability — see MetricRegressionDojo, which this class
    delegates every method to unchanged. Only __init__ differs: it
    fixes metric_name to PickSideboardRateMetric.name (never a
    re-declared string literal — this pairing can't drift) and
    resolves a default metrics_path/loss so the common case needs no
    extra wiring.
    """

    def __init__(
        self,
        expansion: str,
        format_code: str,
        train_ratio: float,
        validate_ratio: float,
        rng_seed: int,
        metrics_path: Path | None = None,
        loss: Loss[torch.Tensor, float] | None = None,
    ) -> None:
        """Construct a dojo that regresses against one (expansion, format_code)'s
        pick_sideboard_rate values.

        Inputs:
            expansion: 17lands expansion code this dojo trains
                against — which specific DraftMetricScanner run's
                output to read.
            format_code: 17lands format code, paired with expansion.
            train_ratio: fraction of eligible cards assigned to the
                train pool.
            validate_ratio: fraction of eligible cards assigned to the
                validate pool. The remainder goes to test.
            rng_seed: seed for this dojo's own shuffling/sampling, for
                reproducible splits and batches.
            metrics_path: which parquet file to read
                pick_sideboard_rate rows from. Defaults to
                DraftMetricScanner.default_output_path(expansion,
                format_code) (this project's conventional location)
                when not given.
            loss: the injected loss computation this dojo delegates to.
                Defaults to MSELoss() when not given — the only
                sensible default for a scalar-regression dojo.
        Output: none (constructor).
        Side effects: none — metrics_path is not read until
            prepare_splits is called.
        Exceptions: raises ValueError if train_ratio + validate_ratio
            is not strictly less than 1, or either ratio is not in
            (0, 1) (delegated from MetricRegressionDojo.__init__).

        Example:
            >>> dojo = PickSideboardRateDojo(
            ...     expansion="MSH",
            ...     format_code="PremierDraft",
            ...     train_ratio=0.8,
            ...     validate_ratio=0.1,
            ...     rng_seed=0,
            ... )
        """
        super().__init__(
            metrics_path=metrics_path
            or DraftMetricScanner.default_output_path(expansion, format_code),
            metric_name=PickSideboardRateMetric.name,
            train_ratio=train_ratio,
            validate_ratio=validate_ratio,
            loss=loss or MSELoss(),
            rng_seed=rng_seed,
        )
