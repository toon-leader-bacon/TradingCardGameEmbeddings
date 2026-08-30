"""Thin MetricRegressionDojo wrapper for game_data_metrics' own
OpeningHandWinRateMetric.

Lives under game_data_dojo/ — replay_data_metrics has a distinct
OpeningHandWinRateMetric of its own (same metric_name string,
different class/file, different raw source CSV), wrapped by the
same-named class one directory over at
replay_data_dojo/opening_hand_win_rate_dojo.py. The per-source
directory (mirroring data_refinement/seventeenlands/'s own
game_data_metrics/ vs. replay_data_metrics/ split) is what disambiguates
the two — same precedent as the two source Metric classes themselves,
which already share this exact name across modules.

See average_pick_number_dojo.py's own docstring for the rationale this
otherwise mirrors exactly — this class exists purely for
discoverability, not new logic.
"""

from pathlib import Path

import torch

from src.data_refinement.seventeenlands.game_data_metrics.metric_scanner import (
    MetricScanner,
)
from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.opening_hand_win_rate import (  # noqa: E501 -- module path is one unbreakable dotted identifier
    OpeningHandWinRateMetric,
)
from src.dojos.losses.loss import Loss
from src.dojos.losses.mse_loss import MSELoss
from src.dojos.seventeenlands.metric_regression_dojo import MetricRegressionDojo


class OpeningHandWinRateDojo(MetricRegressionDojo):
    """Regresses a card's embedding against game_data_metrics'
    opening_hand_win_rate.

    Not a new capability — see MetricRegressionDojo, which this class
    delegates every method to unchanged. Only __init__ differs: it
    fixes metric_name to OpeningHandWinRateMetric.name (this game_data_
    metrics variant, never a re-declared string literal — this pairing
    can't drift) and resolves a default metrics_path/loss so the common
    case needs no extra wiring.
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
        game_data_metrics opening_hand_win_rate values.

        Inputs:
            expansion: 17lands expansion code this dojo trains
                against — which specific MetricScanner run's output to
                read.
            format_code: 17lands format code, paired with expansion.
            train_ratio: fraction of eligible cards assigned to the
                train pool.
            validate_ratio: fraction of eligible cards assigned to the
                validate pool. The remainder goes to test.
            rng_seed: seed for this dojo's own shuffling/sampling, for
                reproducible splits and batches.
            metrics_path: which parquet file to read
                opening_hand_win_rate rows from. Defaults to
                MetricScanner.default_output_path(expansion,
                format_code) (this project's conventional game_data_
                metrics location) when not given.
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
            >>> dojo = OpeningHandWinRateDojo(
            ...     expansion="MSH",
            ...     format_code="PremierDraft",
            ...     train_ratio=0.8,
            ...     validate_ratio=0.1,
            ...     rng_seed=0,
            ... )
        """
        super().__init__(
            metrics_path=metrics_path or MetricScanner.default_output_path(expansion, format_code),
            metric_name=OpeningHandWinRateMetric.name,
            train_ratio=train_ratio,
            validate_ratio=validate_ratio,
            loss=loss or MSELoss(),
            rng_seed=rng_seed,
        )
