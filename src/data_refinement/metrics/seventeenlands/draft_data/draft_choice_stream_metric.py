"""Template Method base for the draft_data row streams: one output row
per single pick, the pack options shown and the option taken (see
draft_data/README.md).

A vectorized Metric[DraftDataChunk] and a RowStreamMetric
(../sliced_metric.py). Per chunk it writes every kept row in one
ParquetBuilder.write_columns() call; list columns (the pack's options,
the pool) are built from a zone's present matrix with np.nonzero, never
row by row.

Output columns: draft_id, pack_number, pick_number, pool_uuids when
INCLUDES_POOL, then pack_option_uuids and pick_uuid (null when the
pick's name matched no card). A subclass fixes OUTPUT_STEM, and
PoolConditionedPickMetric sets INCLUDES_POOL.

PICKTWO ROWS ARE SKIPPED: a two-card pick is not a one-of-N choice, so
a row with a second pick writes nothing.
"""

from pathlib import Path
from typing import Any, ClassVar, Literal, Sequence

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk import (
    DraftDataChunk,
)
from src.data_refinement.seventeenlands.zone_counts import ZoneCounts
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.data_retrieval.seventeenlands.refs import DataType

PACK_OPTIONS_COLUMN = "pack_option_uuids"
PICK_COLUMN = "pick_uuid"
POOL_COLUMN = "pool_uuids"


class DraftChoiceStreamMetric:
    """One single-pick row -> its pack options and the option taken.

    Satisfies the Metric[DraftDataChunk] Protocol (../../metric.py) and
    RowStreamMetric (../sliced_metric.py) structurally.
    """

    FAMILY: ClassVar[DataType] = DataType.DRAFT
    OUTPUT_STEM: ClassVar[str]
    LABEL_COLUMN: ClassVar[str] = PICK_COLUMN
    IS_ROW_STREAM: ClassVar[Literal[True]] = True
    # Whether to write the pool so far (pool_uuids) before the options
    INCLUDES_POOL: ClassVar[bool] = False

    def __init__(
        self, version_metadata: MetricVersionMetadata, output_path: Path
    ) -> None:
        """Open the output for streaming.

        Inputs:
            version_metadata: the CardBinder version this run reads,
                stamped onto the output.
            output_path: this CSV's partition path.
        Output: none (constructor).
        Side effects: creates output_path's parent directories; opens
            output_path for writing (truncating it) through a
            ParquetBuilder held open until finalize().
        Exceptions: whatever ParquetBuilder raises opening output_path.
        """
        self._output_path = output_path
        schema = schema_with_version_metadata(self._output_schema(), version_metadata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(output_path, schema)

    def accumulate(self, chunk: DraftDataChunk) -> None:
        """Write one output row per single-pick row of chunk.

        Inputs: chunk.
        Output: none.
        Side effects: writes the chunk's single-pick rows to the open
            ParquetBuilder (none if every row is a PickTwo pick).
        Exceptions: none expected.

        Example:
            >>> metric = PackToPickChoiceSetMetric(version_metadata, partition_path)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        rows = ~chunk.picks.is_pick_two

        # The scalars, the subclass's list columns, then the choice
        columns: dict[str, Sequence[Any] | npt.NDArray[np.generic]] = {
            "draft_id": chunk.draft_id[rows],
            "pack_number": chunk.pack_number[rows],
            "pick_number": chunk.pick_number[rows],
        }
        if self.INCLUDES_POOL:
            columns[POOL_COLUMN] = _present_uuid_lists(chunk.pool, rows)
        columns[PACK_OPTIONS_COLUMN] = _present_uuid_lists(chunk.pack, rows)
        columns[PICK_COLUMN] = chunk.picks.first_uuids[rows]
        self._writer.write_columns(columns)

    def finalize(self) -> Path:
        """Flush and close the writer. Idempotent.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder.
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/draft_data/pack_to_pick_choice_set/KTK/TradDraft.parquet')
        """
        self._writer.close()
        return self._output_path

    def _output_schema(self) -> pa.Schema:
        """The output schema (without metadata): scalars, the pool when
        INCLUDES_POOL, options, pick.

        Inputs: none. Output: pa.Schema.
        Side effects: none. Exceptions: none.
        """
        uuid_list = pa.list_(pa.string())
        return pa.schema(
            [
                ("draft_id", pa.string()),
                ("pack_number", pa.int64()),
                ("pick_number", pa.int64()),
                *([(POOL_COLUMN, uuid_list)] if self.INCLUDES_POOL else []),
                (PACK_OPTIONS_COLUMN, uuid_list),
                (PICK_COLUMN, pa.string()),
            ]
        )


def _present_uuid_lists(zone: ZoneCounts, rows: npt.NDArray[np.bool_]) -> pa.ListArray:
    """For each kept row, the uuids (as str) of zone's present columns,
    in column (header) order: one list per row.

    Inputs: zone, rows (bool (rows,), which rows to keep).
    Output: pa.ListArray of strings, one list per kept row (empty list
        for a row with nothing present).
    Side effects: none. Exceptions: none.
    """
    present = zone.present()[rows]
    kept_rows, columns = np.nonzero(present)  # row-major: header order per row

    # One list per row: offsets from each row's present count
    offsets = np.zeros(present.shape[0] + 1, np.int32)
    np.cumsum(np.bincount(kept_rows, minlength=present.shape[0]), out=offsets[1:])
    column_uuids = pa.array([str(uuid) for uuid in zone.card_uuids], pa.string())
    values = column_uuids.take(pa.array(columns, pa.int64()))
    return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), values)
