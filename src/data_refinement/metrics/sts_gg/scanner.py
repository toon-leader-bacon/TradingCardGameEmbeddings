"""Drives a shared read pass over sts_gg's runs.jsonl across multiple
Metric instances at once, so metrics sharing this raw source don't
each re-read the file.

See plans/sts_gg_metrics.md's scan_runs_jsonl section for the
failure-isolation contract this implements - deliberately looser than
legacy/sts_gg/deck_outcome_metric.py's own scan_runs_jsonl (which
drives exactly one metric and lets a failure propagate): one metric's
bug must not silently stop a different, still-healthy metric in the
same list from seeing the rest of the file.

DECK BOX IS NOT THIS FILE'S CONCERN: a metric MAY take a DeckBox in
its own constructor and write into it during accumulate() (see
plans/sts_gg_metrics.md and ascension_prediction_metric.py) - this
function only ever calls accumulate()/finalize() on already-
constructed Metric[dict] instances, so it has no involvement in
constructing, injecting, or saving that box. See scan_runs_jsonl()'s
own docstring Example for what the calling driver does around it.
"""

import json
import logging
from pathlib import Path

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
        accumulate()/finalize() failure, rather than raising.
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
        ...     Path("data/metrics/sts_gg/deck_box.jsonl"),
        ...     GameId.SLAY_THE_SPIRE_2,
        ... )  # this function never does this - the driver's own job
    """
    with open(raw_path, "r", encoding="utf-8") as raw_file:
        for line in raw_file:
            if not line.strip():
                continue
            row = json.loads(line)
            # Feed this row to every metric, isolating one metric's
            # accumulate() failure from the rest of the list.
            for metric in metrics:
                _accumulate_isolated(metric, row)

    # Finalize every metric, isolating one metric's finalize()
    # failure from the rest of the list.
    for metric in metrics:
        _finalize_isolated(metric)


def _accumulate_isolated(metric: Metric[dict], row: dict) -> None:
    """Call metric.accumulate(row), logging (not raising) on failure.

    Private helper - single consumer is scan_runs_jsonl(). Isolating
    this per call is what lets one bad metric in the list not corrupt
    another metric's otherwise-fine scan.

    Inputs:
        metric: the Metric[dict] to drive.
        row: one parsed JSON line to feed it.
    Output: none.
    Side effects: whatever metric.accumulate() does on success; on
        failure, emits one logging.exception() call instead of
        propagating.
    Exceptions: none - every exception from metric.accumulate() is
        caught and logged here.
    """
    try:
        metric.accumulate(row)
    except Exception:
        _logger.exception(
            "scan_runs_jsonl: %r raised from accumulate() on one row - "
            "skipping just that row for this metric, continuing the scan "
            "for every other metric",
            metric,
        )


def _finalize_isolated(metric: Metric[dict]) -> None:
    """Call metric.finalize(), logging (not raising) on failure.

    Private helper - single consumer is scan_runs_jsonl(). Mirrors
    _accumulate_isolated()'s isolation for the finalize phase.

    Inputs:
        metric: the Metric[dict] to finalize.
    Output: none.
    Side effects: whatever metric.finalize() does on success; on
        failure, emits one logging.exception() call instead of
        propagating.
    Exceptions: none - every exception from metric.finalize() is
        caught and logged here.
    """
    try:
        metric.finalize()
    except Exception:
        _logger.exception(
            "scan_runs_jsonl: %r raised from finalize() - its output may "
            "be missing or incomplete, but every other metric still "
            "finalizes normally",
            metric,
        )
