"""Writing and memory-mapping single-batch Arrow IPC files.

Shared by snapshots/ and the per-run caches built over them
(DeckSplitIndex): the one place that knows such a file is uncompressed
Arrow IPC holding exactly one record batch (so every column is one
contiguous buffer a NumPy view can wrap), written under a temporary name
and renamed into place. Its errors are neutral; each caller says what a
missing or malformed file means for it.
"""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa

# Suffix of the temporary file a write goes to before its rename
_TEMPORARY_SUFFIX = ".partial"


def write_table_atomically(table: pa.Table, path: Path) -> None:
    """Write table to path as an uncompressed Arrow IPC file with one
    record batch, under a temporary name renamed into place, so a crash
    never leaves a file that looks complete.

    Inputs: table (its schema carries the stamp), path.
    Output: none.
    Side effects: creates path's parent directory; writes a temporary
        file beside path, then replaces path with it.
    Exceptions: OSError on a write failure (the temporary file is
        removed).

    Example:
        >>> write_table_atomically(table, directory / "cards.arrow")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + _TEMPORARY_SUFFIX)

    # Exactly one record batch, even for an empty table, each column contiguous
    single_batch = pa.RecordBatch.from_arrays(
        [column.combine_chunks() for column in table.columns], schema=table.schema
    )
    try:
        with pa.OSFile(str(temporary_path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_batch(single_batch)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def open_mapped_table(path: Path) -> pa.Table:
    """Memory-map path and read its table without copying.

    Inputs: path.
    Output: pa.Table whose buffers point into the mapped file (the table
        keeps the map alive).
    Side effects: maps the file.
    Exceptions: FileNotFoundError if path does not exist; ValueError if
        the file is not Arrow IPC or does not hold exactly one record
        batch.

    Example:
        >>> table = open_mapped_table(directory / "cards.arrow")
    """
    if not path.exists():
        raise FileNotFoundError(f"no Arrow file at {path}")

    source = pa.memory_map(str(path), "r")
    try:
        reader = pa.ipc.open_file(source)
    except pa.ArrowInvalid as error:
        raise ValueError(f"{path} is not an Arrow IPC file") from error

    # Validate: one batch, so each column is one buffer
    if reader.num_record_batches != 1:
        raise ValueError(
            f"{path} holds {reader.num_record_batches} record batches, not 1"
        )
    return reader.read_all()


def fixed_width_bytes_of(column: pa.ChunkedArray, width: int) -> np.ndarray:
    """A single-chunk fixed_size_binary(width) column's values as a
    read-only uint8 view of shape (len, width), straight from its data
    buffer (no per-row pyarrow scalar).

    Inputs: column (from a table open_mapped_table read), width.
    Output: np.ndarray uint8 (len(column), width).
    Side effects: none.
    Exceptions: ValueError if the column has more than one chunk.

    Example:
        >>> fixed_width_bytes_of(table.column("nocab_uuid"), 16)[0].tobytes()
    """
    if column.num_chunks > 1:
        raise ValueError(f"expected one chunk, found {column.num_chunks}")
    if len(column) == 0:
        return np.empty((0, width), dtype=np.uint8)
    chunk = column.chunk(0)
    data = np.frombuffer(chunk.buffers()[1], dtype=np.uint8)
    start = chunk.offset * width
    return data[start : start + len(chunk) * width].reshape(len(chunk), width)
