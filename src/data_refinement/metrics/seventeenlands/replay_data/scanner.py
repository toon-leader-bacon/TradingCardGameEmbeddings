"""Drives a shared, per-row read pass over one 17lands replay_data CSV
across multiple Metric[dict] instances at once - see
plans/replay_data_metrics.md's scan_replay_csv component.

Structural copy of draft_data/scanner.py's scan_draft_csv()/
game_data/scanner.py's scan_game_csv(): chunking here is purely an
I/O-efficiency detail internal to this function - every
Metric.accumulate() call still receives one row at a time, never a
whole chunk.

DOES NOT BUILD ReplayCardColumns: each metric builds its own
ReplayCardColumns (replay_card_columns.py) internally, from the
(card_binder, header, source_game) it was itself constructed with -
this function never touches ReplayCardColumns or card_binder at all,
it only ever drives already-constructed Metric[dict] instances over
rows.
"""

import logging
from pathlib import Path

import pandas as pd

from src.data_refinement.metrics.metric import Metric

_logger = logging.getLogger(__name__)


def scan_replay_csv(
    raw_csv_path: Path, metrics: list[Metric[dict]], chunk_size: int = 100_000
) -> None:
    """Drive every metric in `metrics` over every row of raw_csv_path,
    then finalize all of them.

    Inputs:
        raw_csv_path: path to a 17lands replay_data CSV (e.g.
            data/raw/17lands/replay_data/MSH.PremierDraft.csv).
        metrics: every Metric[dict] to drive over this one read pass -
            each already constructed with this same raw_csv_path's
            (card_binder, header, source_game) (see module docstring).
        chunk_size: rows per pandas.read_csv chunk - an I/O-efficiency
            knob only; does not change what any metric receives (still
            one row at a time).
    Output: none.
    Side effects: reads raw_csv_path once, in chunks; calls
        accumulate() on every metric for every row, then finalize() on
        every metric. Logs loudly (via the stdlib logging module) on
        any per-metric accumulate()/finalize() failure, rather than
        raising - isolates one metric's bug from every other metric in
        the list, same contract as draft_data/game_data's scanners.
    Exceptions: raises if raw_csv_path doesn't exist or isn't parsable
        as CSV - only a per-metric accumulate()/finalize() failure is
        caught and isolated, not a raw-file-level failure.

    Example:
        >>> header = pd.read_csv(raw_csv_path, nrows=0).columns
        >>> metrics = [
        ...     AverageTurnCastMetric(card_binder, header, GameId.MTG),
        ... ]
        >>> scan_replay_csv(raw_csv_path, metrics)
    """
    # low_memory=False (deviation from draft_data/game_data's scanners):
    # replay_data's ~2500 mostly-sparse per-turn columns otherwise get
    # inconsistent per-internal-chunk dtype inference from pandas' C
    # parser - ReplayCardColumns.arena_uuids() already normalizes
    # either shape, but reading in one dtype pass per chunk avoids the
    # DtypeWarning and keeps behavior deterministic across runs.
    for chunk in pd.read_csv(raw_csv_path, chunksize=chunk_size, low_memory=False):
        for row in chunk.to_dict(orient="records"):
            for metric in metrics:
                _accumulate_isolated(metric, row)

    for metric in metrics:
        _finalize_isolated(metric)


def _accumulate_isolated(metric: Metric[dict], row: dict) -> None:
    """Call metric.accumulate(row), logging (not raising) on failure.

    Private helper - single consumer is scan_replay_csv(). Mirrors
    draft_data/game_data scanners' own _accumulate_isolated() exactly.

    Inputs:
        metric: the Metric[dict] to drive.
        row: one parsed CSV row to feed it.
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
            "scan_replay_csv: %r raised from accumulate() on one row - "
            "skipping just that row for this metric, continuing the scan "
            "for every other metric.",
            metric,
        )


def _finalize_isolated(metric: Metric[dict]) -> None:
    """Call metric.finalize(), logging (not raising) on failure.

    Private helper - single consumer is scan_replay_csv(). Mirrors
    draft_data/game_data scanners' own _finalize_isolated() exactly.

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
            "scan_replay_csv: %r raised from finalize() - this metric's "
            "output may be missing or incomplete.",
            metric,
        )
