"""Drives one checkpointed, single-pass scan of a 17lands game_data CSV.

See src/data_refinement/README.md for this container's scope and
this container's own README for the full contract this was built
against.

MetricScanner's constructor requires a source_game parameter:
resolving card columns (see column_lookup.py) requires one, since
CardBinder is generic across GameId, so something has to say which
game's cards this scan's CSV should be resolved against.

MetricScanner streams raw_csv_path in chunks (pandas — chosen as
sufficient for a single streaming pass, and the natural fit for
join/groupby-heavy multi-card metrics
anticipated later), driving every active Metric's accumulate() per
chunk, so N active metrics still cost one file pass, not N. Checkpoints
(rows processed + every metric's Memento state) are written every
checkpoint_every_n_chunks chunks, so a crash/restart resumes instead
of rescanning from row 0.

**Shared machinery (this session):** the checkpoint read/write and
per-chunk streaming loop are no longer implemented here — they were
byte-for-byte identical to DraftMetricScanner/ReplayMetricScanner's own
copies (see tmp/REFACTOR.md §1), so this class now composes
MetricCheckpoint (metric_checkpoint.py) and accumulate_over_chunks()
(metric_scan_loop.py) instead. Column resolution (_find_card_columns())
is unchanged — it's the one genuinely pipeline-specific piece.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.column_lookup import (
    ColumnResolution,
    find_card_columns,
)
from src.data_refinement.seventeenlands.game_data_metrics.metric import Metric
from src.data_refinement.seventeenlands.metric_checkpoint import MetricCheckpoint
from src.data_refinement.seventeenlands.metric_scan_loop import accumulate_over_chunks
from src.data_refinement.seventeenlands.metric_writer import write_metric_results
from src.schema.game_id import GameId


@dataclass(frozen=True)
class MetricScanResult:
    """What one MetricScanner.scan() call produced.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    output_path: Path
    unresolved_column_names: list[str]  # card names in the CSV header
    # CardBinder couldn't resolve


class MetricScanner:
    """Drives a checkpointed single pass over one game_data CSV.

    Single-consumer-per-scan — one MetricScanner instance handles one
    (raw_csv_path, metrics) combination; a different CSV or a different
    set of active metrics gets its own MetricScanner instance.
    """

    DEFAULT_OUTPUT_DIR: ClassVar[Path] = Path("data/final/metrics/17lands/game")
    DEFAULT_OUTPUT_NAME: ClassVar[str] = "{expansion}.{format_code}.parquet"

    def __init__(
        self,
        raw_csv_path: Path,
        card_binder: CardBinder,
        metrics: list[Metric],
        output_path: Path,
        checkpoint_path: Path,
        source_game: GameId,
        checkpoint_every_n_chunks: int = 10,
    ) -> None:
        """
        Inputs:
            raw_csv_path: path to a 17lands game_data CSV (e.g.
                data/raw/17lands/game_data/MSH.PremierDraft.csv).
            card_binder: registry to resolve card columns against.
            metrics: every Metric to run over this CSV in one pass.
                Each must already be fully constructed (e.g. a
                WinRateMetric already knows the expansion/format it
                should stamp onto its MetricResults — see
                metrics/win_rate/win_rate.py — since that's caller-level
                wiring, not MetricScanner's concern).
            output_path: where finished MetricResult rows are written,
                as parquet.
            checkpoint_path: where an in-progress scan's checkpoint is
                written/read. Does not need to exist yet.
            source_game: which game raw_csv_path's cards belong to —
                see this module's docstring for why this parameter
                exists (a deviation from the originally approved
                skeleton).
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

    @staticmethod
    def default_output_path(expansion: str, format_code: str) -> Path:
        """The conventional output_path for one (expansion, format_code) scan.

        A recommended default, not an enforced requirement — output_path
        stays a required constructor parameter; this exists so a caller
        can compute the same conventional path this project already
        uses elsewhere, instead of re-typing the string (see e.g.
        scripts/compute_seventeenlands_metrics.py).

        Inputs:
            expansion: 17lands expansion code.
            format_code: 17lands format code.
        Output: DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME, formatted
            with expansion/format_code (e.g.
            data/final/metrics/17lands/game/MSH.PremierDraft.parquet).
        Side effects: none — purely a path computation.
        Exceptions: none.

        Example:
            >>> MetricScanner.default_output_path("MSH", "PremierDraft")
            PosixPath('data/final/metrics/17lands/game/MSH.PremierDraft.parquet')
        """
        return MetricScanner.DEFAULT_OUTPUT_DIR / MetricScanner.DEFAULT_OUTPUT_NAME.format(
            expansion=expansion, format_code=format_code
        )

    def scan(self) -> MetricScanResult:
        """Resolve columns, stream the CSV, checkpoint, and write results.

        Composed of:
            1. _find_card_columns() — resolves raw_csv_path's header
               against card_binder (see column_lookup.py).
            2. Constructs a MetricCheckpoint(self.checkpoint_path) and
               calls its restore() to resume from a prior scan, if one
               exists (returns 0 otherwise, leaving every metric's
               fresh, empty state untouched).
            3. accumulate_over_chunks() (metric_scan_loop.py) — streams
               raw_csv_path, resolving each chunk via a callback that
               always returns step 1's already-computed
               column_resolution.resolved (the header is resolved once,
               not per chunk), driving every metric's accumulate() and
               writing periodic checkpoints. See that function's own
               docstring for the exact cumulative-rows_processed
               correctness requirement (getting it wrong corrupts a
               subsequent resume).
            4. write_metric_results() (metric_writer.py) — calls
               finalize() on every metric and writes the merged
               MetricResult rows to output_path as parquet.
            5. Constructs and returns the MetricScanResult (this is the
               one place this object gets built), then deletes
               checkpoint_path (a completed scan has nothing left to
               resume).

        Inputs: none (uses constructor-supplied state).
        Output: a MetricScanResult naming output_path and every card
            name from raw_csv_path's header that couldn't be resolved.
        Side effects: reads raw_csv_path and (if present)
            checkpoint_path; writes checkpoint_path periodically during
            the scan; writes output_path at the end; deletes
            checkpoint_path on successful completion.
        Exceptions: raises on failure to read raw_csv_path, on failure
            to write checkpoint_path or output_path, or whatever a
            metric's accumulate()/finalize()/load_state() raises.

        Example:
            >>> scanner = MetricScanner(
            ...     Path("data/raw/17lands/game_data/MSH.PremierDraft.csv"),
            ...     card_binder,
            ...     [WinRateMetric(expansion="MSH", format_code="PremierDraft")],
            ...     Path("data/final/metrics/17lands/game/MSH.PremierDraft.parquet"),
            ...     Path("data/final/metrics/17lands/game/MSH.PremierDraft.checkpoint.json"),
            ...     GameId.MTG,
            ... )
            >>> result = scanner.scan()
        """
        column_resolution = self._find_card_columns()
        checkpoint = MetricCheckpoint(self.checkpoint_path)
        rows_processed, _ = checkpoint.restore(self.metrics)

        rows_processed, _ = accumulate_over_chunks(
            self.raw_csv_path,
            self.metrics,
            lambda _chunk: column_resolution.resolved,
            checkpoint,
            rows_processed,
            self.checkpoint_every_n_chunks,
            "MetricScanner chunks",
        )

        write_metric_results(self.metrics, self.output_path)
        result = MetricScanResult(
            output_path=self.output_path,
            unresolved_column_names=column_resolution.unresolved_names,
        )
        checkpoint.delete()
        return result

    def _find_card_columns(self) -> ColumnResolution:
        """Resolve raw_csv_path's header against card_binder.

        Private helper — single consumer is scan(). Reads only
        raw_csv_path's header row (never the full file) before
        delegating to column_lookup.find_card_columns().

        Inputs: none (uses self.raw_csv_path, self.card_binder,
            self.source_game).
        Output: a ColumnResolution splitting the header's card names
            into resolved CardColumnSets and unresolved names.
        Side effects: reads raw_csv_path's header row.
        Exceptions: raises if raw_csv_path doesn't exist or has no
            readable header row.
        """
        header = pd.read_csv(self.raw_csv_path, nrows=0).columns.tolist()
        return find_card_columns(header, self.card_binder, self.source_game)
