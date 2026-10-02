from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import pyarrow.parquet as pq

from src.data_refinement.metrics.parquet_builder import (
    ParquetBuilder,
)

_SCHEMA = pa.schema([("id", pa.string()), ("count", pa.int64())])


class TestWriteRow:
    def test_flushes_automatically_at_batch_size(self, tmp_path: Path) -> None:
        path = tmp_path / "out.parquet"
        writer = ParquetBuilder(path, _SCHEMA, batch_size=2)

        # Exactly one batch's worth - the automatic flush at the batch
        # boundary should leave nothing for close() to still write.
        writer.write_row({"id": "a", "count": 1})
        writer.write_row({"id": "b", "count": 2})
        writer.close()

        table = pq.read_table(path)
        assert table.to_pylist() == [
            {"id": "a", "count": 1},
            {"id": "b", "count": 2},
        ]

    def test_writes_more_than_one_batch_worth_of_rows_in_order(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.parquet"
        writer = ParquetBuilder(path, _SCHEMA, batch_size=3)

        for i in range(7):
            writer.write_row({"id": str(i), "count": i})
        writer.close()

        table = pq.read_table(path)
        assert table.num_rows == 7
        assert [row["id"] for row in table.to_pylist()] == [str(i) for i in range(7)]


class TestWriteColumns:
    def test_writes_whole_columns_after_buffered_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "out.parquet"
        writer = ParquetBuilder(path, _SCHEMA, batch_size=100)

        writer.write_row({"id": "a", "count": 1})
        writer.write_columns(
            {"id": np.array(["b", "c"], object), "count": np.array([2, 3])}
        )
        writer.write_row({"id": "d", "count": 4})
        writer.close()

        assert [row["id"] for row in pq.read_table(path).to_pylist()] == [
            "a",
            "b",
            "c",
            "d",
        ]

    def test_zero_rows_write_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "out.parquet"
        writer = ParquetBuilder(path, _SCHEMA, batch_size=100)

        writer.write_columns({"id": [], "count": []})
        writer.close()

        assert pq.read_table(path).num_rows == 0

    def test_a_missing_column_raises(self, tmp_path: Path) -> None:
        writer = ParquetBuilder(tmp_path / "out.parquet", _SCHEMA)

        with pytest.raises(KeyError):
            writer.write_columns({"id": ["a"]})
        writer.close()


class TestClose:
    def test_flushes_a_partial_final_batch(self, tmp_path: Path) -> None:
        path = tmp_path / "out.parquet"
        writer = ParquetBuilder(path, _SCHEMA, batch_size=100)

        writer.write_row({"id": "a", "count": 1})
        writer.close()

        assert pq.read_table(path).num_rows == 1

    def test_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "out.parquet"
        writer = ParquetBuilder(path, _SCHEMA, batch_size=100)
        writer.write_row({"id": "a", "count": 1})

        writer.close()
        writer.close()

        assert pq.read_table(path).num_rows == 1

    def test_produces_a_valid_empty_file_with_no_rows_written(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.parquet"
        writer = ParquetBuilder(path, _SCHEMA, batch_size=100)

        writer.close()

        assert pq.read_table(path).num_rows == 0

    def test_preserves_schema_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "out.parquet"
        schema = _SCHEMA.with_metadata({b"game": b"mtg"})
        writer = ParquetBuilder(path, schema, batch_size=100)

        writer.write_row({"id": "a", "count": 1})
        writer.close()

        assert pq.read_schema(path).metadata[b"game"] == b"mtg"
