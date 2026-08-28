"""Drives one checkpointed, single-pass scan of a 17lands replay_data CSV.

See src/data_refinement/README.md for this container's scope and
../draft_data_metrics/draft_metric_scanner.py's DraftMetricScanner for the
sibling driver this now mirrors most closely (see arena_id_cache.py's
docstring for why): both containers' card identity lives in per-ROW
data rather than the CSV header, so both own one cache instance for
the whole scan (ArenaIdCache here, PickNameCache there), constructed
once in scan() and called once per chunk, rather than resolving a
fixed header up front the way game_data_metrics/metric_scanner.py's
MetricScanner does.

ReplayMetricScanner streams raw_csv_path in chunks (pandas, same choice
and rationale as MetricScanner), driving every active ReplayMetric's
accumulate() per chunk, so N active metrics still cost one file pass,
not N. Checkpoints (rows processed + unresolved Arena IDs so far +
every metric's Memento state) are written every
checkpoint_every_n_chunks chunks, so a crash/restart resumes instead of
rescanning from row 0.

**Shared machinery (this session):** the checkpoint read/write and
per-chunk streaming loop are no longer implemented here — they were
identical in shape to MetricScanner/DraftMetricScanner's own copies
(see tmp/REFACTOR.md §1), so this class now composes MetricCheckpoint
(metric_checkpoint.py) and accumulate_over_chunks() (metric_scan_loop.py)
instead. The one piece that's genuinely specific to this pipeline is
the UNION-of-unresolved-Arena-IDs bookkeeping, expressed as a
checkpoint_extra closure passed to accumulate_over_chunks() — see
scan()'s docstring below for exactly how that closure works.
"""

from dataclasses import dataclass
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.metric_checkpoint import MetricCheckpoint
from src.data_refinement.seventeenlands.metric_scan_loop import accumulate_over_chunks
from src.data_refinement.seventeenlands.metric_writer import write_metric_results
from src.data_refinement.seventeenlands.replay_data_metrics.arena_id_cache import (
    ArenaIdCache,
)
from src.data_refinement.seventeenlands.replay_data_metrics.replay_metric import (
    ReplayMetric,
)
from src.schema.game_id import GameId


@dataclass(frozen=True)
class ReplayMetricScanResult:
    """What one ReplayMetricScanner.scan() call produced.

    unresolved_arena_ids is a SET of distinct Arena ID strings, not a
    running count — matches MetricScanResult.unresolved_column_names
    and DraftMetricScanResult.unresolved_pick_names' pattern of NAMING
    what failed to resolve, not just counting failures. Bounded by the
    number of distinct cards ever referenced in the scan (a few
    hundred), not by how many rows/cells reference them.

    The container type itself (frozenset, not those two siblings' list)
    is a deliberate deviation, not full parity with either precedent:
    ReplayMetricScanner's checkpoint resume needs to UNION a restored
    set with whatever a fresh ArenaIdCache accumulates during the
    resumed portion of the scan (see scan()'s docstring, step 3) — set
    semantics make that trivial; a list would need manual deduplication
    on every merge. frozen also matches this already-frozen=True
    dataclass.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    output_path: Path
    unresolved_arena_ids: frozenset[str]


class ReplayMetricScanner:
    """Drives a checkpointed single pass over one replay_data CSV.

    Single-consumer-per-scan — one ReplayMetricScanner instance handles
    one (raw_csv_path, metrics) combination; a different CSV or a
    different set of active metrics gets its own ReplayMetricScanner
    instance.
    """

    def __init__(
        self,
        raw_csv_path: Path,
        card_binder: CardBinder,
        metrics: list[ReplayMetric],
        output_path: Path,
        checkpoint_path: Path,
        source_game: GameId,
        checkpoint_every_n_chunks: int = 10,
    ) -> None:
        """
        Inputs:
            raw_csv_path: path to a 17lands replay_data CSV (e.g.
                data/raw/17lands/replay_data/MSH.PremierDraft.csv).
            card_binder: registry to resolve hand-cell Arena IDs
                against.
            metrics: every ReplayMetric to run over this CSV in one
                pass. Each must already be fully constructed (e.g.
                already knows the expansion/format it should stamp
                onto its MetricResults — see
                metrics/binary_trigger_rate/opening_hand_win_rate.py —
                since that's caller-level wiring, not
                ReplayMetricScanner's concern).
            output_path: where finished MetricResult rows are written,
                as parquet.
            checkpoint_path: where an in-progress scan's checkpoint is
                written/read. Does not need to exist yet.
            source_game: which game raw_csv_path's cards belong to.
            checkpoint_every_n_chunks: how many chunks between
                checkpoint writes.
        Output: none (constructor).
        Side effects: none — no I/O happens until scan() is called.
        Exceptions: none.
        """
        self.raw_csv_path = raw_csv_path
        self.card_binder = card_binder
        self.metrics = metrics
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.source_game = source_game
        self.checkpoint_every_n_chunks = checkpoint_every_n_chunks

    def scan(self) -> ReplayMetricScanResult:
        """Stream the CSV, resolving hands and checkpointing, then write results.

        Composed of:
            1. Constructs one ArenaIdCache(self.card_binder,
               self.source_game) for the whole scan — mirrors
               DraftMetricScanner's PickNameCache construction; this is
               the "resolve" step's entire setup, no up-front header
               resolution. The cache's cache and unresolved_ids
               accumulate across every chunk of THIS scan() call, but a
               resumed scan starts a fresh instance (see the restored
               extra payload note below for why that's still correct).
            2. Constructs a MetricCheckpoint(self.checkpoint_path) and
               calls its restore() to resume from a prior scan, if one
               exists. The restored extra payload's
               "unresolved_arena_ids" list (or [] if absent/no
               checkpoint) is captured as restored_unresolved_arena_ids
               — a frozenset, NOT seeded back into the fresh ArenaIdCache
               instance from step 1. Correctness only requires the
               FINAL reported set (this checkpoint's ids unioned with
               whatever the fresh cache accumulates from here) to be
               complete, not that the cache instance itself remembers
               earlier misses — the checkpoint_extra closure in step 3
               performs exactly that union, every time it's called.
            3. accumulate_over_chunks() (metric_scan_loop.py) — streams
               raw_csv_path, resolving each chunk via
               cache.get_hands(chunk), driving every metric's
               accumulate(), and writing periodic checkpoints. Passes a
               checkpoint_extra closure that computes
               restored_unresolved_arena_ids | cache.unresolved_ids
               fresh on every call — cache.unresolved_ids is itself
               already cumulative across every get_hands() call made to
               this instance so far, so this union is the complete
               total-to-date without any separate manual accumulation.
            4. write_metric_results() (metric_writer.py) — calls
               finalize() on every metric and writes the merged
               MetricResult rows to output_path as parquet, using the
               FINAL extra value accumulate_over_chunks() returned
               (computed via one more checkpoint_extra() call after the
               loop ends — see that function's own docstring for why
               this is an in-memory recomputation, not an extra disk
               write).
            5. Constructs and returns the ReplayMetricScanResult (this
               is the one place this object gets built), then deletes
               checkpoint_path (a completed scan has nothing left to
               resume).

        Inputs: none (uses constructor-supplied state).
        Output: a ReplayMetricScanResult naming output_path and every
            distinct Arena ID across the whole scan that couldn't be
            resolved.
        Side effects: reads raw_csv_path and (if present)
            checkpoint_path; writes checkpoint_path periodically during
            the scan; writes output_path at the end; deletes
            checkpoint_path on successful completion.
        Exceptions: raises on failure to read raw_csv_path, on failure
            to write checkpoint_path or output_path, or whatever a
            metric's accumulate()/finalize()/load_state() raises.

        Example:
            >>> scanner = ReplayMetricScanner(
            ...     Path("data/raw/17lands/replay_data/MSH.PremierDraft.csv"),
            ...     card_binder,
            ...     [OpeningHandWinRateMetric(expansion="MSH", format_code="PremierDraft")],
            ...     Path("data/final/metrics/17lands/MSH.PremierDraft.replay.parquet"),
            ...     Path("data/final/metrics/17lands/MSH.PremierDraft.replay.checkpoint.json"),
            ...     GameId.MTG,
            ... )
            >>> result = scanner.scan()
        """
        cache = ArenaIdCache(self.card_binder, self.source_game)
        checkpoint = MetricCheckpoint(self.checkpoint_path)
        rows_processed, extra = checkpoint.restore(self.metrics)
        restored_unresolved_arena_ids = frozenset(extra.get("unresolved_arena_ids", []))

        def checkpoint_extra() -> dict:
            current = restored_unresolved_arena_ids | cache.unresolved_ids
            return {"unresolved_arena_ids": sorted(current)}

        rows_processed, final_extra = accumulate_over_chunks(
            self.raw_csv_path,
            self.metrics,
            lambda chunk: cache.get_hands(chunk),
            checkpoint,
            rows_processed,
            self.checkpoint_every_n_chunks,
            "ReplayMetricScanner chunks",
            checkpoint_extra=checkpoint_extra,
        )
        unresolved_arena_ids = frozenset(final_extra.get("unresolved_arena_ids", []))

        write_metric_results(self.metrics, self.output_path)
        result = ReplayMetricScanResult(
            output_path=self.output_path,
            unresolved_arena_ids=unresolved_arena_ids,
        )
        checkpoint.delete()
        return result
