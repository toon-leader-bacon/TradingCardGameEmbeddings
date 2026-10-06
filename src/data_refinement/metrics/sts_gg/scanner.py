"""Drives a shared read pass over sts_gg's runs.jsonl across multiple
Metric instances at once, so metrics sharing this raw source don't
each re-read the file.

The failure-isolation contract this implements is deliberately looser
than
legacy/sts_gg/deck_outcome_metric.py's own scan_runs_jsonl (which
drives exactly one metric and lets a failure propagate): one metric's
bug must not silently stop a different, still-healthy metric in the
same list from seeing the rest of the file.

DECK BOX IS NOT THIS FILE'S CONCERN: a metric MAY take a DeckBox in
its own constructor and write into it during accumulate() (see
sts_gg/README.md and ascension_prediction_metric.py) - this
function only ever calls accumulate()/finalize() on already-
constructed Metric[dict] instances, so it has no involvement in
constructing, injecting, or saving that box. See scan_runs_jsonl()'s
own docstring Example for what the calling driver does around it.
"""

import json
import logging
from functools import partial
from pathlib import Path

from tqdm import tqdm

from src.data_refinement.metrics.isolated_call import call_isolated
from src.data_refinement.metrics.metric import Metric

_logger = logging.getLogger(__name__)


def scan_runs_jsonl(raw_path: Path, metrics: list[Metric[dict]]) -> None:
    """Drive every metric in `metrics` over every line of raw_path,
    then finalize all of them.

    Inputs:
        raw_path: path to a runs.jsonl file (one JSON object per line
            - e.g. data/raw/sts_gg/runs.jsonl).
        metrics: every Metric[dict] to drive over this one read pass.
            Each metric's own accumulate()/finalize() failures are
            isolated from the others (see module docstring).
    Output: none.
    Side effects: reads raw_path once; calls accumulate() on every
        metric for every line, then finalize() on every metric. Logs
        loudly (via the stdlib logging module) on any per-metric
        accumulate()/finalize() failure, rather than raising. Prints a
        tqdm progress bar to stderr, sized against raw_path's byte size
        (not its line count, which isn't known up front without a
        separate full read).
    Exceptions: raises if raw_path doesn't exist, or if any line isn't
        valid JSON - only a per-metric accumulate()/finalize() failure
        is caught and isolated, not a raw-file-level failure.

    Example:
        >>> deck_box = DeckBox()  # metrics-private - never the published box
        >>> metrics = [
        ...     CardUpgradeRateMetric(card_binder),  # single-card: no DeckBox
        ...     AscensionPredictionMetric(card_binder, deck_box),
        ... ]
        >>> scan_runs_jsonl(Path("data/raw/sts_gg/runs.jsonl"), metrics)
        >>> deck_box.save(
        ...     Path("data/metrics/sts_gg/deck_box.db"),
        ...     GameId.SLAY_THE_SPIRE_2,
        ...     card_binder.version_for(GameId.SLAY_THE_SPIRE_2),
        ... )  # this function never does this - the driver's own job
    """
    total_bytes = raw_path.stat().st_size
    with open(raw_path, "r", encoding="utf-8") as raw_file, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=f"scan_runs_jsonl: {raw_path.name}",
    ) as progress:
        for line in raw_file:
            progress.update(len(line.encode("utf-8")))
            if not line.strip():
                continue
            row = json.loads(line)
            # Feed this row to every metric, isolating one metric's
            # accumulate() failure from the rest of the list.
            for metric in metrics:
                call_isolated(
                    _logger,
                    metric,
                    "accumulate() on one row",
                    partial(metric.accumulate, row),
                )

    # Finalize every metric, isolating one metric's finalize()
    # failure from the rest of the list.
    for metric in metrics:
        call_isolated(_logger, metric, "finalize()", metric.finalize)
