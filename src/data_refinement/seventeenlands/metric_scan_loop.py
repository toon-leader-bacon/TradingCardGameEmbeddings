"""Shared per-chunk scan loop for every *MetricScanner.

Confirmed identical TODAY between MetricScanner and ReplayMetricScanner's
scan() bodies: the CSV-chunk-streaming call itself
(pd.read_csv(raw_csv_path, chunksize=..., skiprows=...)), the totaled
tqdm progress-bar construction and per-metric advance, the
checkpoint-every-N-chunks trigger, and the cumulative rows_processed
increment. DraftMetricScanner does NOT have this today — it currently
uses a bare, un-totaled tqdm bar with no per-metric advance and no
count_data_rows() call at all (see tmp/REFACTOR.md §2's "Unexplained
sibling inconsistency," still open as of this skeleton). Adopting this
shared loop is what BRINGS DraftMetricScanner up to the other two's
behavior — not a description of code that already exists in all three.

The ONE per-chunk step that genuinely differs per pipeline is computing
that chunk's `resolved` value to hand to every metric's accumulate() —
game_data_metrics resolves once, up front, and reuses the same value
every chunk (expressed here as a callback that ignores its chunk
argument and returns a precomputed constant — no repeated work happens,
since the closure itself does nothing but return an already-computed
value); draft_data_metrics/replay_data_metrics resolve fresh per chunk
via their own cache. Captured as the `resolve` callback parameter below.

Adopting this loop for all three scanners is what satisfies the
separately-agreed progress-bar unification (option 1 from that
conversation: bring DraftMetricScanner up to match, not simplify the
other two down) — DraftMetricScanner gets the same totaled,
per-metric-advance bar as its two siblings by using this same shared
loop, rather than needing its own hand-written fix.

_CHUNK_SIZE_ROWS moves here from being duplicated as a private constant in
all three *MetricScanner files — this is now the one place chunk size is
decided.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Protocol, TypeVar

import pandas as pd
from tqdm import tqdm

from src.data_refinement.seventeenlands.csv_row_count import count_data_rows
from src.data_refinement.seventeenlands.metric_checkpoint import MetricCheckpoint
from src.data_refinement.seventeenlands.scannable_metric import ScannableMetric

_CHUNK_SIZE_ROWS = 100_000

TResolved = TypeVar("TResolved", contravariant=True)


class ChunkAccumulator(ScannableMetric, Protocol[TResolved]):
    """ScannableMetric (name/finalize/save_state/load_state) plus the one
    additional method this loop calls per chunk: accumulate(). Extends
    ScannableMetric rather than standing alone, because this loop passes
    `metrics` straight through to checkpoint.write() (see
    accumulate_over_chunks() below) — every concrete
    Metric/DraftMetric/ReplayMetric already satisfies this combined
    Protocol structurally, since accumulate()'s signature is the only
    thing that differs between those three per-pipeline Protocols.
    """

    def accumulate(self, chunk: pd.DataFrame, resolved: TResolved) -> None: ...


def accumulate_over_chunks(
    raw_csv_path: Path,
    metrics: Sequence[ChunkAccumulator[TResolved]],
    resolve: Callable[[pd.DataFrame], TResolved],
    checkpoint: MetricCheckpoint,
    rows_already_processed: int,
    checkpoint_every_n_chunks: int,
    progress_desc: str,
    checkpoint_extra: Callable[[], dict] | None = None,
) -> tuple[int, dict]:
    """Stream raw_csv_path in chunks, resolving + accumulating +
    checkpointing identically for every *MetricScanner.

    Composed of: count_data_rows(raw_csv_path) (csv_row_count.py) to
    size a totaled tqdm progress bar (advanced once per metric's
    accumulate() call per chunk, matching the prior MetricScanner/
    ReplayMetricScanner-only behavior — see this module's docstring for
    why DraftMetricScanner now gets this too); for each chunk (skipping
    rows_already_processed rows, resuming a prior scan): computes
    resolved = resolve(chunk), calls every metric's
    accumulate(chunk, resolved); writes a checkpoint via
    checkpoint.write() every checkpoint_every_n_chunks chunks, passing
    checkpoint_extra() as that write's extra payload if checkpoint_extra
    was given (e.g. ReplayMetricScanner supplies a closure computing the
    current UNION of restored + newly-discovered unresolved Arena IDs,
    recomputed fresh at every PERIODIC checkpoint write — see
    replay_data_metrics/README.md for why that union must be
    recomputed each time, not just once). After the loop ends,
    checkpoint_extra() (if given) is called exactly once more to compute
    this function's own return value — this final call is an IN-MEMORY
    recomputation only, not an extra disk write: no checkpoint.write()
    happens after the loop, since the caller deletes the checkpoint file
    once it finishes writing final results, making one last write-then-
    immediately-delete pointless.

    Inputs:
        raw_csv_path: path to the CSV to stream.
        metrics: every metric to drive accumulate() on, once per
            chunk.
        resolve: called once per chunk, before any metric's
            accumulate(), producing that chunk's resolved value.
        checkpoint: where to write periodic checkpoints during this
            call. This function does NOT call checkpoint.restore() or
            checkpoint.delete() itself — the caller resolves the
            starting rows_already_processed via restore() before
            calling this, and calls delete() after, once results are
            also written.
        rows_already_processed: starting row offset (0 for a fresh
            scan, or checkpoint.restore()'s returned count when
            resuming).
        checkpoint_every_n_chunks: how many chunks between checkpoint
            writes.
        progress_desc: the tqdm bar's desc string (e.g.
            "DraftMetricScanner chunks") — kept caller-supplied so
            each Scanner's bar still names itself distinctly.
        checkpoint_extra: optional zero-argument callback. Called once
            per PERIODIC checkpoint write, to compute that write's
            extra payload, plus exactly once more after the loop ends
            to compute this function's own returned extra value (an
            in-memory recomputation, not an additional disk write —
            see this function's docstring above). None if this
            pipeline has no extra state to checkpoint
            (game_data_metrics, draft_data_metrics today).
    Output: a tuple of (final cumulative rows_processed across the
        whole scan, the final checkpoint_extra() value if given, else
        {}) — the caller uses the second element to construct its own
        ScanResult's pipeline-specific "unresolved" field.
    Side effects: reads raw_csv_path; writes checkpoint.checkpoint_path
        periodically via checkpoint.write(). Does not write
        output_path or delete the checkpoint — those remain the
        caller's responsibility, after this function returns.
    Exceptions: raises on failure to read raw_csv_path, on failure to
        write a checkpoint, or whatever a metric's accumulate() or
        resolve() raises.

    Example:
        >>> rows_processed, extra = accumulate_over_chunks(
        ...     raw_csv_path, metrics,
        ...     lambda chunk: cache.get_uuids(chunk["pick"]),
        ...     checkpoint, 0, 10, "DraftMetricScanner chunks",
        ... )
    """
    skiprows = (
        range(1, rows_already_processed + 1) if rows_already_processed > 0 else None
    )
    chunk_iterator = pd.read_csv(
        raw_csv_path, chunksize=_CHUNK_SIZE_ROWS, skiprows=skiprows
    )

    total_data_rows = count_data_rows(raw_csv_path)
    remaining_rows = max(total_data_rows - rows_already_processed, 0)
    total_chunks = -(-remaining_rows // _CHUNK_SIZE_ROWS)  # ceil div
    steps_per_chunk = max(len(metrics), 1)

    rows_processed = rows_already_processed
    chunks_since_checkpoint = 0
    with tqdm(
        total=total_chunks * steps_per_chunk,
        desc=progress_desc,
        unit="metric-chunk",
    ) as progress:
        for chunk in chunk_iterator:
            resolved = resolve(chunk)
            for metric in metrics:
                metric.accumulate(chunk, resolved)
                progress.update(1)
            if not metrics:
                progress.update(1)

            rows_processed += len(chunk)
            chunks_since_checkpoint += 1
            if chunks_since_checkpoint >= checkpoint_every_n_chunks:
                extra = checkpoint_extra() if checkpoint_extra is not None else None
                checkpoint.write(rows_processed, metrics, extra=extra)
                chunks_since_checkpoint = 0

    final_extra = checkpoint_extra() if checkpoint_extra is not None else {}
    return rows_processed, final_extra
