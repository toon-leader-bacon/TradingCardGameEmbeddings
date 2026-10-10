"""Survey every catalog dojo's baseline_loss, for choosing the loss-weighting
baseline floor (plans/baseline_normalized_loss_weighting.md, step 1).

For each catalog key, builds the dojo on real data, pulls a few TRAIN
batches and prints one CSV row: the baseline's min / median / max over the
batches, how many batches had no usable baseline, the mean raw loss against
random embeddings, and the mean seconds one baseline_loss call took. A key
that cannot be built or yields nothing prints an error row instead.

Usage (from the project root):

    PYTHONPATH=. venv/Scripts/python.exe scripts/survey_dojo_baselines.py > baselines.csv
    PYTHONPATH=. venv/Scripts/python.exe scripts/survey_dojo_baselines.py --keys a b
"""

import argparse
import logging
import statistics
import time
from dataclasses import dataclass
from itertools import islice

import torch

from src.dojos.dojo import BatchBudget, Dojo
from src.dojos.loss.regression_objective import (
    RegressionLossKind,
    RegressionObjective,
)
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.training.dojo_catalog import (
    DOJO_CATALOG,
    CardShelf,
    DojoBuildContext,
    build_dojos,
)
from src.training.preflight import random_embeddings_like
from scripts.run_training import card_cost

_CSV_HEADER = (
    "key,baseline_min,baseline_median,baseline_max,unusable_batches,batches,"
    "mean_raw_loss,baseline_seconds,error"
)


@dataclass(frozen=True)
class BaselineSurvey:
    """One dojo's baseline behavior over a handful of TRAIN batches.

    baselines: baseline_loss of each batch that had one.
    unusable_batches: batches whose baseline_loss raised ValueError.
    mean_raw_loss: mean compute_loss against random embeddings.
    baseline_seconds: mean wall time of one baseline_loss call.
    """

    baselines: list[float]
    unusable_batches: int
    mean_raw_loss: float
    baseline_seconds: float


def survey_dojo(
    dojo: Dojo, budget: BatchBudget, batches: int, size: int
) -> BaselineSurvey:
    """Pull up to `batches` TRAIN batches from dojo and measure its baselines.

    Inputs: dojo (built), budget (BatchBudget), batches (how many to pull),
        size (card_embedding_size for the random embeddings).
    Output: BaselineSurvey.
    Side effects: advances the dojo's TRAIN stream and mod random state.
    Exceptions: ValueError if the dojo yields no batch; whatever the dojo's
        compute_loss raises.

    Example:
        >>> survey_dojo(dojo, budget, 10, 32).baselines[:2]
        [1.0, 1.0]
    """
    baselines: list[float] = []
    raw_losses: list[float] = []
    unusable = 0
    spent = 0.0
    for batch in islice(dojo.batches(Split.TRAIN, budget), batches):
        embeddings = random_embeddings_like(batch.inputs, size)
        raw_losses.append(dojo.compute_loss(embeddings, batch).item())
        started = time.perf_counter()
        try:
            baselines.append(dojo.baseline_loss(batch))
        except ValueError:
            unusable += 1
        spent += time.perf_counter() - started
    if not raw_losses:
        raise ValueError("no TRAIN batches")
    return BaselineSurvey(
        baselines, unusable, statistics.fmean(raw_losses), spent / len(raw_losses)
    )


def format_row(key: str, survey: BaselineSurvey) -> str:
    """One CSV row for a surveyed key (see _CSV_HEADER).

    Inputs: key (catalog key), survey. Output: str. Side effects: none.
    Exceptions: none.
    """
    if survey.baselines:
        low = f"{min(survey.baselines):.6g}"
        middle = f"{statistics.median(survey.baselines):.6g}"
        high = f"{max(survey.baselines):.6g}"
    else:
        low = middle = high = ""
    total = len(survey.baselines) + survey.unusable_batches
    return (
        f"{key},{low},{middle},{high},{survey.unusable_batches},{total},"
        f"{survey.mean_raw_loss:.6g},{survey.baseline_seconds:.3g},"
    )


def main() -> int:
    """Survey the chosen keys (all by default), printing a CSV to stdout.

    Inputs: command line (--keys, --batches, --max-batch-cost, --card-embedding-size).
    Output: 0.
    Side effects: builds dojos (may create split files); prints.
    Exceptions: none for a single key's failure (printed as an error row).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", nargs="*", default=sorted(DOJO_CATALOG))
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--max-batch-cost", type=int, default=256)
    parser.add_argument("--card-embedding-size", type=int, default=32)
    parser.add_argument(
        "--regression-loss",
        type=RegressionLossKind,
        choices=list(RegressionLossKind),
        default=RegressionLossKind.MSE,
        help="loss the regression dojos are built with (their baselines differ)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    torch.manual_seed(0)

    context = DojoBuildContext(
        shelf=CardShelf(),
        holdout=HoldoutSpec.no_holdout(),
        card_embedding_size=args.card_embedding_size,
        rng_seed=0,
        regression_objective=RegressionObjective.for_kind(args.regression_loss),
    )
    budget = BatchBudget(args.max_batch_cost, card_cost)
    print(_CSV_HEADER, flush=True)
    # One key at a time, so a failure or a big dojo costs only itself
    for key in args.keys:
        try:
            dojo = build_dojos([key], context)[0]
            survey = survey_dojo(dojo, budget, args.batches, args.card_embedding_size)
            print(format_row(key, survey), flush=True)
        except Exception as error:
            message = f"{type(error).__name__}: {error}".replace(",", ";")
            print(f"{key},,,,,,,,{message[:150]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
