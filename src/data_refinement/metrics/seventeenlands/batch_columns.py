"""Reading typed numpy columns out of one pyarrow CSV record batch:
the steps both 17lands chunk parsers (game_data's GameDataChunkParser,
draft_data's DraftDataChunkParser) share.

Card count columns are read as float32 (some exports write "1.0") and
narrowed to int16 here; a null count is 0. A scalar column holding a
null is rejected, naming the column and row.
"""

from collections.abc import Sequence
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.zone_counts import ZoneCounts

# Read type for every card count column (narrowed by read_zone_counts)
CARD_COUNT_TYPE = pa.float32()


def read_zone_counts(
    batch: pa.RecordBatch, columns: Sequence[tuple[str, UUID]]
) -> ZoneCounts:
    """One zone's (rows x columns) int16 count matrix; nulls become 0.

    Inputs: batch, columns (header column, card uuid) for this zone, in
        header order.
    Output: ZoneCounts (empty card_uuids and a (rows, 0) matrix when
        columns is empty).
    Side effects: none. Exceptions: none.

    Example:
        >>> read_zone_counts(batch, parser_columns)
    """
    if not columns:
        return ZoneCounts(card_uuids=(), counts=np.zeros((batch.num_rows, 0), np.int16))

    # One numpy column per matched header column; a null cell means 0
    arrays = [
        batch.column(column).fill_null(0).to_numpy(zero_copy_only=False)
        for column, _ in columns
    ]
    counts = np.column_stack(arrays).astype(np.int16)
    return ZoneCounts(card_uuids=tuple(uuid for _, uuid in columns), counts=counts)


def raise_on_null(batch: pa.RecordBatch, column: str) -> None:
    """Raise if batch's column holds a null.

    Inputs: batch, column. Output: none. Side effects: none.
    Exceptions: ValueError naming column and the first null's offset.

    Example:
        >>> raise_on_null(batch, "draft_id")
    """
    values = batch.column(column)
    if values.null_count == 0:
        return
    first_null = int(np.flatnonzero(values.is_null().to_numpy(zero_copy_only=False))[0])
    raise ValueError(
        f"column {column!r} is null at batch row {first_null} "
        f"({values.null_count} nulls in this batch)"
    )


def read_column(
    batch: pa.RecordBatch, column: str, dtype: type[np.generic]
) -> npt.NDArray:
    """batch's column as a numpy array of dtype (a null string reads as
    None; callers reject nulls first where they matter).

    Inputs: batch, column, dtype (e.g. np.bool_, np.int64, np.object_
        for strings). Output: shape (rows,).
    Side effects: none. Exceptions: KeyError if column is missing.

    Example:
        >>> read_column(batch, "pack_number", np.int64)
    """
    values = batch.column(column).to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype)
