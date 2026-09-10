"""Drives a shared read pass over play_gwent's guides.jsonl across
multiple Metric instances at once, so metrics sharing this raw source
don't each re-read the file.

Deliberately its own copy of sts_gg/scanner.py's shape rather than a
shared/centralized driver - see this project's convention of
deferring cross-container deck/metric-plumbing dedup to a later,
comprehensive pass rather than centralizing on first duplication.

DECK BOX IS NOT THIS FILE'S CONCERN: a metric writes into its own
DeckBox during accumulate() (see deck_card_mask_metric.py) - this
function only ever calls accumulate()/finalize() on already-
constructed Metric[dict] instances.
"""

import json
import logging
from pathlib import Path

from src.data_refinement.metrics.metric import Metric

_logger = logging.getLogger(__name__)


def scan_guides_jsonl(raw_path: Path, metrics: list[Metric[dict]]) -> None:
    """Drive every metric in `metrics` over every line of raw_path,
    then finalize all of them.

    Inputs:
        raw_path: path to a guides.jsonl file (one JSON guide object
            per line - e.g. data/raw/play_gwent/guides.jsonl).
        metrics: every Metric[dict] to drive over this one read pass.
            Each metric's own accumulate()/finalize() failures are
            isolated from the others (see module docstring).
    Output: none.
    Side effects: reads raw_path once; calls accumulate() on every
        metric for every line, then finalize() on every metric. Logs
        loudly (via the stdlib logging module) on any per-metric
        accumulate()/finalize() failure, rather than raising.
    Exceptions: raises if raw_path doesn't exist - a per-metric
        accumulate()/finalize() failure is caught and isolated, not a
        raw-file-level failure. A line that isn't valid JSON is
        silently skipped (matches leader_deck_counts.py's convention
        for this same raw source).

    Example:
        >>> box = DeckBox.load([Path("data/final/decks/gwent.jsonl")])
        >>> metrics = [LeaderMaskedFromDeckMetric(card_lookup, box)]
        >>> scan_guides_jsonl(Path("data/raw/play_gwent/guides.jsonl"), metrics)
        >>> box.save(Path("data/final/decks/gwent.jsonl"), GameId.GWENT)
    """
    with open(raw_path, "r", encoding="utf-8") as raw_file:
        for line in raw_file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
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

    Private helper - single consumer is scan_guides_jsonl().

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
            "scan_guides_jsonl: %r raised from accumulate() on one row - "
            "skipping just that row for this metric, continuing the scan "
            "for every other metric",
            metric,
        )


def _finalize_isolated(metric: Metric[dict]) -> None:
    """Call metric.finalize(), logging (not raising) on failure.

    Private helper - single consumer is scan_guides_jsonl().

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
            "scan_guides_jsonl: %r raised from finalize() - its output may "
            "be missing or incomplete, but every other metric still "
            "finalizes normally",
            metric,
        )
