"""Drives one streaming read pass over a 17lands CSV across every
chunk metric at once: scan_chunked_csv(), shared by the game_data and
draft_data families (see their READMEs).

The file is streamed with pyarrow.csv.open_csv in record batches. Each
batch is parsed exactly once, by the ChunkParser the driver built for
this CSV, and that one chunk is handed to every metric. The scanner
never sees the CardBinder or a DeckBox: card matching lives in the
parser, and a deck-input metric received its DeckBox in its own
constructor (the driver saves that box afterwards).

Failure isolation is per (metric, chunk): one metric raising on a chunk
is logged and skips that chunk for that metric only; every other metric
still gets it. A metric that raises partway through accumulate() can be
left with that chunk half-tallied, so a logged accumulate() failure
means that metric's output for this CSV must not be trusted.
"""

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol, Sequence, TypeVar

import pyarrow as pa
import pyarrow.csv as pa_csv
from tqdm import tqdm

from src.data_refinement.metrics.isolated_call import FAILURE_MARKER, call_isolated
from src.data_refinement.metrics.metric import Metric

ChunkT = TypeVar("ChunkT")
ChunkT_co = TypeVar("ChunkT_co", covariant=True)

_logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 64 << 20  # bytes of CSV per record batch


class UnsupportedCsvLayout(ValueError):
    """Raised by a ChunkParser factory for a CSV whose column layout its
    family deliberately does not read (an older 17lands export missing
    what the metrics need). The driver skips such a CSV rather than
    counting it as a failure; any other ValueError is a failure."""


class ChunkParser(Protocol[ChunkT_co]):
    """One CSV's column layout, turning each record batch into one
    typed chunk (GameDataChunkParser, DraftDataChunkParser)."""

    def needed_columns(self) -> list[str]:
        """The columns to read, in header order."""
        ...

    def column_types(self) -> dict[str, pa.DataType]:
        """The read type of each needed column."""
        ...

    def parse(self, batch: pa.RecordBatch) -> ChunkT_co:
        """One record batch as a chunk; ValueError on a bad batch."""
        ...


def scan_chunked_csv(
    raw_csv_path: Path,
    metrics: Sequence[Metric[ChunkT]],
    parser: ChunkParser[ChunkT],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> None:
    """Drive every metric over every chunk of raw_csv_path, then
    finalize all of them.

    Inputs:
        raw_csv_path: a 17lands CSV (e.g.
            data/raw/17lands/game_data/KTK.TradDraft.csv).
        metrics: every metric to drive over this one read pass.
        parser: built by the driver from this same CSV's header.
        block_size: CSV bytes per record batch (an I/O knob; does not
            change any metric's output).
    Output: none.
    Side effects: reads raw_csv_path once; calls accumulate(chunk) on
        every metric per batch, then finalize() on every metric; logs
        (does not raise) any one metric's failure, naming the metric,
        the CSV and the chunk, then one summary line per failing metric
        at the end; shows a tqdm byte progress bar on stderr.
    Exceptions: raises if raw_csv_path is missing or not parsable as
        CSV; ValueError naming the CSV and chunk if parser.parse()
        rejects a batch (a null scalar). These are file-level failures,
        not one metric's: they stop the whole family run.

    Example:
        >>> parser = GameDataChunkParser.from_header(header, binder, GameId.MTG)
        >>> scan_chunked_csv(path, metrics, parser)
    """
    # Read only what the parser needs, with card counts as small ints
    convert_options = pa_csv.ConvertOptions(
        include_columns=parser.needed_columns(),
        column_types=parser.column_types(),
    )
    read_options = pa_csv.ReadOptions(block_size=block_size)

    failures = _FailureTally(raw_csv_path.name)
    chunk_count = 0

    # Stream batches; parse each once and hand it to every metric
    with open(raw_csv_path, "rb") as raw_file, _byte_progress(raw_csv_path) as progress:
        reader = pa_csv.open_csv(
            raw_file, read_options=read_options, convert_options=convert_options
        )
        for chunk_index, batch in enumerate(reader):
            chunk = _parse_or_raise(parser, batch, raw_csv_path, chunk_index)
            chunk_count += 1
            step = f"accumulate() on {raw_csv_path.name} chunk {chunk_index}"
            for metric in metrics:
                ok = call_isolated(
                    _logger, metric, step, partial(metric.accumulate, chunk)
                )
                failures.record_accumulate(metric, ok)
            progress.update(raw_file.tell() - progress.n)

    # Finalize every metric, isolating one failure from the rest
    for metric in metrics:
        ok = call_isolated(
            _logger, metric, f"finalize() on {raw_csv_path.name}", metric.finalize
        )
        failures.record_finalize(metric, ok)

    failures.log_summary(chunk_count)


def _parse_or_raise(
    parser: ChunkParser[ChunkT],
    batch: pa.RecordBatch,
    raw_csv_path: Path,
    chunk_index: int,
) -> ChunkT:
    """parser.parse(batch), with a rejected batch's error naming the CSV
    and chunk so the bad row can be found in the file.

    Inputs: parser, batch, raw_csv_path, chunk_index. Output: the chunk.
    Side effects: none.
    Exceptions: ValueError, prefixed with the CSV name and chunk index.
    """
    try:
        return parser.parse(batch)
    except ValueError as error:
        raise ValueError(f"{raw_csv_path.name} chunk {chunk_index}: {error}") from error


def _byte_progress(raw_csv_path: Path) -> tqdm:
    """A tqdm bar sized to raw_csv_path's byte size.

    Inputs: raw_csv_path. Output: tqdm (a context manager).
    Side effects: none until used. Exceptions: OSError if the file is
        missing.
    """
    return tqdm(
        total=raw_csv_path.stat().st_size,
        unit="B",
        unit_scale=True,
        desc=f"scan: {raw_csv_path.name}",
    )


@dataclass
class _MetricFailures:
    """One metric's failed steps over one CSV.

    accumulate_failures: chunks whose accumulate() raised.
    finalize_failed: whether finalize() raised (it runs once per CSV).
    """

    accumulate_failures: int = 0
    finalize_failed: bool = False


class _FailureTally:
    """Each metric's failed steps over one CSV, so the scan ends with one
    loud summary line per failing metric."""

    def __init__(self, csv_name: str) -> None:
        """Start with no failures.

        Inputs: csv_name (named in the summary). Output: none.
        Side effects: none. Exceptions: none.
        """
        self._csv_name = csv_name
        self._failures: dict[str, _MetricFailures] = {}

    def record_accumulate(self, metric: object, ok: bool) -> None:
        """Count one failed accumulate() for metric (success: nothing).

        Inputs: metric, ok (call_isolated's result). Output: none.
        Side effects: updates the tally. Exceptions: none.
        """
        if not ok:
            self._failures_of(metric).accumulate_failures += 1

    def record_finalize(self, metric: object, ok: bool) -> None:
        """Mark metric's finalize() as failed (success: nothing).

        Inputs: metric, ok (call_isolated's result). Output: none.
        Side effects: updates the tally. Exceptions: none.
        """
        if not ok:
            self._failures_of(metric).finalize_failed = True

    def log_summary(self, chunk_count: int) -> None:
        """One ERROR line per failing metric; nothing when all passed.

        Inputs: chunk_count (chunks scanned). Output: none.
        Side effects: logs. Exceptions: none.
        """
        for metric_name, failures in self._failures.items():
            _logger.error(
                "%s summary for %s: %s failed accumulate() on %d of %d chunks%s",
                FAILURE_MARKER,
                self._csv_name,
                metric_name,
                failures.accumulate_failures,
                chunk_count,
                (
                    " and failed finalize() (output missing or untrustworthy)"
                    if failures.finalize_failed
                    else ""
                ),
            )

    def _failures_of(self, metric: object) -> _MetricFailures:
        """metric's record, created on its first failure.

        Inputs: metric. Output: _MetricFailures.
        Side effects: may add a record. Exceptions: none.
        """
        return self._failures.setdefault(repr(metric), _MetricFailures())
