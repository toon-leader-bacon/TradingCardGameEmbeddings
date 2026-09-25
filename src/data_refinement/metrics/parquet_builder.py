"""Buffers per-row writes into fixed-size parquet row groups, instead
of one write_table() call (and row group) per row - see
pool_conditioned_pick_metric.py's and pack_to_pick_choice_set_metric.py's
module docstrings for why one row group per row breaks down on a large
CSV (tens of millions of rows): ParquetWriter accumulates per-row-group
metadata in memory for its whole lifetime, so RAM grows and throughput
collapses well before the file is written (observed killing an actual
run on MSH.PremierDraft.csv, ~40M rows - src/training/TODO.md section C).
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_DEFAULT_BATCH_SIZE = 50_000


class ParquetBuilder:
    """Wraps a pyarrow.parquet.ParquetWriter, batching write_row()
    calls into one write_table() (one row group) per `batch_size`
    rows, rather than the caller writing one row group per row itself.

    Not a Metric (../../metric.py) - a low-level I/O helper any
    streaming metric that writes one output row at a time can use in
    place of driving a ParquetWriter directly.
    """

    def __init__(
        self, path: Path, schema: pa.Schema, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> None:
        """
        Inputs:
            path: where the underlying ParquetWriter writes - same
                truncate-on-open behavior as constructing
                pq.ParquetWriter(path, schema) directly.
            schema: this writer's fixed output schema (metadata
                included, e.g. from schema_with_version_metadata()) -
                every write_row() call must supply exactly these
                column names.
            batch_size: how many buffered rows trigger an automatic
                flush - a few tens of thousands keeps the in-memory
                buffer trivial while cutting row-group count by the
                same factor versus one row group per row (see module
                docstring).
        Output: none (constructor).
        Side effects: opens `path` for writing via ParquetWriter,
            truncating any existing file.
        Exceptions: whatever ParquetWriter raises on failure to open
            path.
        """
        self._schema = schema
        self._batch_size = batch_size
        self._writer = pq.ParquetWriter(path, schema)
        self._buffer: dict[str, list[Any]] = {field.name: [] for field in schema}
        self._buffered_rows = 0
        self._closed = False

    def write_row(self, row: dict[str, Any]) -> None:
        """Buffer one row, flushing automatically once `batch_size`
        rows have accumulated.

        Inputs:
            row: one output row, keyed by every column name in this
                writer's schema.
        Output: none.
        Side effects: appends to the in-memory buffer; may flush (see
            _flush()'s own side effects).
        Exceptions: KeyError if `row` is missing one of this writer's
            schema's column names.

        Example:
            >>> writer = ParquetBuilder(path, schema)
            >>> writer.write_row({"draft_id": "d1", "pick_uuid": None})
            >>> writer.close()
        """
        for name, values in self._buffer.items():
            values.append(row[name])
        self._buffered_rows += 1
        if self._buffered_rows >= self._batch_size:
            self._flush()

    def close(self) -> None:
        """Flush any buffered rows and close the underlying writer.

        Idempotent: a second call is a no-op.

        Inputs: none.
        Output: none.
        Side effects: writes one final, possibly-partial row group for
            whatever remains buffered, then closes the ParquetWriter.
        Exceptions: whatever ParquetWriter.close() raises.
        """
        if self._closed:
            return
        self._flush()
        self._writer.close()
        self._closed = True

    def _flush(self) -> None:
        """Write the current buffer as one row group and clear it.

        Private helper - called by write_row() at the batch boundary
        and by close() for whatever remains.

        Inputs: none.
        Output: none.
        Side effects: one write_table() call (a no-op if the buffer is
            empty); clears the buffer.
        Exceptions: none expected beyond whatever write_table() raises.
        """
        if self._buffered_rows == 0:
            return
        table = pa.Table.from_pydict(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        for values in self._buffer.values():
            values.clear()
        self._buffered_rows = 0
