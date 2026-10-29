"""Time how long each dojo takes to build TRAIN batches on the CPU alone.

Usage (from project root, ROCm venv):
    PYTHONPATH=. python experiments/2026-10-29_throughput_shakeout/analysis/batch_build_bench.py \
        <run config> --batches 300 [--max-batch-cost N] > results/<name>.csv

Builds the config's dojos exactly as scripts/run_training.py does (binder
mods, dojo mods, holdout, seed), then pulls --batches TRAIN batches from
each dojo in turn, single-threaded, with no model, feature cache or GPU.
This is the work the trainer's prefetch worker does before the batch's
texts reach the feature cache. Serialization and cache lookups are not
included, so the time is a lower bound on the worker's cost per batch.

Output CSV, one row per dojo: dojo, batches, cards, seconds, ms_per_batch,
ms_per_card, plus the first batch's time separately (it can include
one-off loading).
"""

import argparse
import csv
import sys
import time
from itertools import islice
from pathlib import Path

import torch

from scripts.run_training import card_cost
from src.dojos.dojo import BatchBudget
from src.dojos.loss.regression_objective import RegressionObjective
from src.schema.splits import Split
from src.schema.type_hints import iter_cards
from src.training.dojo_catalog import DOJO_CATALOG, build_dojos
from src.training.dojo_recipes.card_shelf import CardShelf
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext
from src.training.run_config import parse_run_config, read_config_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--max-batch-cost", type=int, default=None)
    args = parser.parse_args()

    config = parse_run_config(read_config_document(args.config), tuple(DOJO_CATALOG))
    torch.manual_seed(config.plan.seed)
    context = DojoBuildContext(
        shelf=CardShelf(binder_mods=config.binder_mods),
        holdout=config.plan.holdout,
        card_embedding_size=config.model.card_embedding_size,
        rng_seed=config.plan.seed,
        mod_overrides=config.mod_overrides,
        staple_thresholds=config.staple_thresholds,
        regression_objective=RegressionObjective.for_kind(config.regression_loss),
        default_augmentation=config.default_augmentation,
    )
    started = time.perf_counter()
    dojos = build_dojos(config.dojo_names, context)
    print(f"built {len(dojos)} dojos in {time.perf_counter() - started:.1f} s", file=sys.stderr)

    cost = args.max_batch_cost or config.limits.max_batch_cost
    budget = BatchBudget(cost, card_cost)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["dojo", "max_batch_cost", "batches", "cards", "first_batch_s", "seconds",
         "ms_per_batch", "ms_per_card"]
    )
    for dojo in dojos:
        stream = dojo.batches(Split.TRAIN, budget)
        first_started = time.perf_counter()
        next(stream)
        first = time.perf_counter() - first_started

        # The timed batches, after the first
        cards = 0
        count = 0
        started = time.perf_counter()
        for batch in islice(stream, args.batches):
            cards += sum(1 for _ in iter_cards(batch.inputs))
            count += 1
        seconds = time.perf_counter() - started
        writer.writerow(
            [dojo.name, cost, count, cards, f"{first:.3f}", f"{seconds:.3f}",
             f"{1000 * seconds / count:.2f}", f"{1000 * seconds / max(cards, 1):.4f}"]
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
