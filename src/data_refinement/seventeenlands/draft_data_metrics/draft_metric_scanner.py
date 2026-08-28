"""Drives one checkpointed, single-pass scan of a 17lands draft_data CSV.

See src/data_refinement/README.md for this container's scope and
game_data_metrics/metric_scanner.py for the sibling this mirrors — same
overall shape (checkpointed streaming pass, parquet output). The one
real structural difference: "resolving" here means something different
from game_data_metrics.

game_data's MetricScanner resolves the CSV's HEADER once, up front,
producing a fixed list of CardColumnSet objects reused unchanged
across every chunk. draft_data's card reference for the two metrics
built so far (average_pick_number, pick_sideboard_rate) is the 'pick'
column's per-ROW VALUE, not anything in the header — so there is
nothing to resolve once up front. Instead, DraftMetricScanner owns one
PickNameCache for the whole scan and calls it once per chunk, on that
chunk's 'pick' column, before handing the resolved result to every
metric's accumulate().

draft_data also has header-based per-card column groups
(pack_card_<name>, pool_<name> — see the real file inspected for this
feature) that neither current metric needs. Building header resolution
for those is explicitly deferred, not attempted here — a future metric
that actually needs them can add that machinery when it exists, rather
than speculatively building it now for nothing to consume.

**Shared machinery (this session):** the checkpoint read/write and
per-chunk streaming loop (including the progress bar) are no longer
implemented here — they were byte-for-byte identical to
MetricScanner/ReplayMetricScanner's own copies once this class adopts
them (see tmp/REFACTOR.md §1), so this class now composes
MetricCheckpoint (metric_checkpoint.py) and accumulate_over_chunks()
(metric_scan_loop.py) instead. This also resolves the previously
unexplained progress-bar inconsistency (see tmp/REFACTOR.md §2):
DraftMetricScanner now gets the same totaled, per-metric-advance tqdm
bar as its two siblings, rather than the bare, un-totaled bar it used
before.
"""

from dataclasses import dataclass
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.draft_data_metrics.draft_metric import (
    DraftMetric,
)
from src.data_refinement.seventeenlands.draft_data_metrics.pick_name_cache import (
    PickNameCache,
)
from src.data_refinement.seventeenlands.metric_checkpoint import MetricCheckpoint
from src.data_refinement.seventeenlands.metric_scan_loop import accumulate_over_chunks
from src.data_refinement.seventeenlands.metric_writer import write_metric_results
from src.schema.game_id import GameId


@dataclass(frozen=True)
class DraftMetricScanResult:
    """What one DraftMetricScanner.scan() call produced.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    output_path: Path
    unresolved_pick_names: list[str]  # 'pick' column values CardBinder
    # couldn't resolve, across the whole scan


class DraftMetricScanner:
    """Drives a checkpointed single pass over one draft_data CSV.

    Single-consumer-per-scan — one DraftMetricScanner instance handles
    one (raw_csv_path, metrics) combination; a different CSV or a
    different set of active metrics gets its own DraftMetricScanner
    instance.
    """

    def __init__(
        self,
        raw_csv_path: Path,
        card_binder: CardBinder,
        metrics: list[DraftMetric],
        output_path: Path,
        checkpoint_path: Path,
        source_game: GameId,
        checkpoint_every_n_chunks: int = 10,
    ) -> None:
        """
        Inputs:
            raw_csv_path: path to a 17lands draft_data CSV (e.g.
                data/raw/17lands/draft_data/MSH.PremierDraft.csv).
            card_binder: registry to resolve 'pick' column values
                against.
            metrics: every DraftMetric to run over this CSV in one
                pass. Each must already be fully constructed (e.g.
                already knows the expansion/format it should stamp
                onto its MetricResults — caller-level wiring, not
                DraftMetricScanner's concern, same convention as
                MetricScanner).
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

    def scan(self) -> DraftMetricScanResult:
        """Stream the CSV, resolving picks per chunk, checkpointing, and
        writing results.

        Composed of:
            1. Constructs one PickNameCache(self.card_binder,
               self.source_game) for the whole scan — this is the
               "resolve" step's entire setup; unlike MetricScanner,
               there is no up-front header resolution to do first (see
               this module's docstring for why).
            2. Constructs a MetricCheckpoint(self.checkpoint_path) and
               calls its restore() to resume from a prior scan, if one
               exists (returns 0 otherwise, leaving every metric's
               fresh, empty state untouched).
            3. accumulate_over_chunks() (metric_scan_loop.py) — streams
               raw_csv_path, resolving each chunk via
               cache.get_uuids(chunk["pick"]), driving every metric's
               accumulate() and writing periodic checkpoints. See that
               function's own docstring for the exact
               cumulative-rows_processed correctness requirement
               (getting it wrong corrupts a subsequent resume).
            4. write_metric_results() (metric_writer.py) — calls
               finalize() on every metric and writes the merged
               MetricResult rows to output_path as parquet.
            5. Constructs and returns the DraftMetricScanResult (this
               is the one place this object gets built), then deletes
               checkpoint_path (a completed scan has nothing left to
               resume).

        Inputs: none (uses constructor-supplied state).
        Output: a DraftMetricScanResult naming output_path and every
            'pick' column value CardBinder couldn't resolve, across
            the whole scan.
        Side effects: reads raw_csv_path and (if present)
            checkpoint_path; writes checkpoint_path periodically
            during the scan; writes output_path at the end; deletes
            checkpoint_path on successful completion.
        Exceptions: raises on failure to read raw_csv_path, on
            failure to write checkpoint_path or output_path, or
            whatever a metric's accumulate()/finalize()/load_state()
            raises.

        Example:
            >>> scanner = DraftMetricScanner(
            ...     Path("data/raw/17lands/draft_data/MSH.PremierDraft.csv"),
            ...     card_binder,
            ...     [AveragePickNumberMetric(expansion="MSH", format_code="PremierDraft")],
            ...     Path("data/final/metrics/17lands/draft/MSH.PremierDraft.parquet"),
            ...     Path("data/final/metrics/17lands/draft/MSH.PremierDraft.checkpoint.json"),
            ...     GameId.MTG,
            ... )
            >>> result = scanner.scan()
        """
        cache = PickNameCache(self.card_binder, self.source_game)
        checkpoint = MetricCheckpoint(self.checkpoint_path)
        rows_processed, _ = checkpoint.restore(self.metrics)

        rows_processed, _ = accumulate_over_chunks(
            self.raw_csv_path,
            self.metrics,
            lambda chunk: cache.get_uuids(chunk["pick"]),
            checkpoint,
            rows_processed,
            self.checkpoint_every_n_chunks,
            "DraftMetricScanner chunks",
        )

        write_metric_results(self.metrics, self.output_path)
        result = DraftMetricScanResult(
            output_path=self.output_path,
            unresolved_pick_names=cache.unresolved_names,
        )
        checkpoint.delete()
        return result
