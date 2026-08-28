"""Shared checkpoint helper for every *MetricScanner — cumulative
rows-processed tracking plus every metric's Memento (save_state/
load_state) state, with one optional slot for a pipeline-specific extra
JSON-serializable payload (e.g. ReplayMetricScanner's unresolved_arena_ids).

Extracted from three near-identical private implementations
(_restore_checkpoint_if_present/_write_checkpoint on MetricScanner,
DraftMetricScanner, and ReplayMetricScanner — see tmp/REFACTOR.md §1),
which differed only in ReplayMetricScanner threading one extra field
through the same shape.

The on-disk JSON checkpoint's "job_states" key is renamed "metric_states"
here (matching this session's Job -> Metric rename) — a safe change since
a checkpoint is a transient, deletable, in-progress-scan artifact, never a
committed data asset; an in-flight checkpoint written before this change
simply won't be resumable (treated as absent, restarting that one scan
from row 0), not silently misread.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from src.data_refinement.seventeenlands.scannable_metric import ScannableMetric


class MetricCheckpoint:
    """Owns the on-disk checkpoint file for one *MetricScanner.scan() call.

    Single-consumer-per-scan — constructed fresh inside each scan()
    call. Holds no state of its own beyond checkpoint_path; all actual
    state lives on disk between calls.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        """
        Inputs:
            checkpoint_path: where this checkpoint is written/read.
                Does not need to exist yet.
        Output: none (constructor).
        Side effects: none — no I/O happens until restore()/write() is
            called.
        Exceptions: none.
        """
        self.checkpoint_path = checkpoint_path

    def restore(self, metrics: Sequence[ScannableMetric]) -> tuple[int, dict]:
        """Restore every metric's state from checkpoint_path, if it exists.

        Every metric in metrics is matched to its saved state by
        metric.name and restored via load_state().

        Inputs:
            metrics: every metric to restore, if a checkpoint exists.
        Output: a tuple of (rows already processed, the extra payload
            last written via write()'s extra parameter, or {} if
            checkpoint_path doesn't exist or was written with no
            extra). If checkpoint_path doesn't exist, returns (0, {})
            and leaves every metric's state untouched — each stays in
            whatever fresh state its own constructor left it in.
        Side effects: reads checkpoint_path if it exists; calls
            load_state() on every metric in metrics if it exists.
        Exceptions: raises if checkpoint_path exists but isn't shaped
            as a checkpoint this class itself would have written (see
            write()), or if metrics contains a metric whose name isn't
            present in the checkpoint's metric_states (e.g. a metric
            added between the crashed scan and this resume).

        Example:
            >>> checkpoint = MetricCheckpoint(Path("scan.checkpoint.json"))
            >>> rows_processed, extra = checkpoint.restore(metrics)
        """
        if not self.checkpoint_path.exists():
            return 0, {}

        state = json.loads(self.checkpoint_path.read_text())
        metric_states = state["metric_states"]
        for metric in metrics:
            metric.load_state(metric_states[metric.name])
        return state["rows_processed"], state.get("extra", {})

    def write(
        self,
        rows_processed: int,
        metrics: Sequence[ScannableMetric],
        extra: dict | None = None,
    ) -> None:
        """Write a checkpoint capturing every metric's current state.

        Overwrites checkpoint_path if it already exists (each call
        replaces the previous checkpoint, not appends).

        Inputs:
            rows_processed: CUMULATIVE count of data rows processed
                since row 0 of the raw CSV — including any rows
                already accounted for by a checkpoint this scan
                resumed from, not just rows read during this call.
                Getting this wrong (writing a session-local count
                instead) would corrupt a subsequent resume: the next
                restore would under-report progress, causing rows to
                be reprocessed and every metric's accumulator
                double-counted for them.
            metrics: every metric whose save_state() gets captured.
            extra: an optional JSON-serializable dict of
                pipeline-specific extra state (e.g.
                {"unresolved_arena_ids": [...]}), round-tripped
                opaquely — this class doesn't interpret its contents,
                only stores and returns it via restore().
        Output: none.
        Side effects: creates checkpoint_path's parent directory if
            missing; writes/overwrites checkpoint_path.
        Exceptions: raises on failure to write checkpoint_path.
        """
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "rows_processed": rows_processed,
            "metric_states": {metric.name: metric.save_state() for metric in metrics},
            "extra": extra if extra is not None else {},
        }
        self.checkpoint_path.write_text(json.dumps(state))

    def delete(self) -> None:
        """Delete checkpoint_path, if it exists.

        Called once a scan completes successfully — a completed scan
        has nothing left to resume.

        Inputs: none.
        Output: none.
        Side effects: deletes checkpoint_path if it exists; no-op if
            it doesn't.
        Exceptions: none expected.
        """
        self.checkpoint_path.unlink(missing_ok=True)
