from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from src.data_refinement.snapshots.arrow_files import (
    fixed_width_bytes_of,
    open_mapped_table,
    write_table_atomically,
)


def _table(rows: int) -> pa.Table:
    schema = pa.schema([("value", pa.int32())], metadata={b"stamp": b"v1"})
    return pa.Table.from_arrays([pa.array(range(rows), type=pa.int32())], schema=schema)


class TestWriteAndOpen:
    def test_round_trips_values_and_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "values.arrow"

        write_table_atomically(_table(5), path)
        table = open_mapped_table(path)

        assert table.column("value").to_pylist() == [0, 1, 2, 3, 4]
        assert table.schema.metadata == {b"stamp": b"v1"}
        assert [p.name for p in path.parent.iterdir()] == ["values.arrow"]

    def test_a_chunked_table_is_written_as_one_batch(self, tmp_path: Path) -> None:
        path = tmp_path / "values.arrow"
        chunked = pa.concat_tables([_table(3), _table(2)])

        write_table_atomically(chunked, path)

        assert open_mapped_table(path).column("value").num_chunks == 1

    def test_an_empty_table_is_one_empty_batch(self, tmp_path: Path) -> None:
        path = tmp_path / "values.arrow"

        write_table_atomically(_table(0), path)

        assert open_mapped_table(path).num_rows == 0

    def test_a_failed_write_leaves_no_partial_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "values.arrow"

        def _fail(*args: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("os.replace", _fail)
        with pytest.raises(OSError, match="disk full"):
            write_table_atomically(_table(3), path)
        assert list(tmp_path.iterdir()) == []


class TestOpenErrors:
    def test_a_missing_file_is_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_mapped_table(tmp_path / "missing.arrow")

    def test_a_non_arrow_file_is_a_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.arrow"
        path.write_bytes(b"not arrow")
        with pytest.raises(ValueError, match="not an Arrow IPC file"):
            open_mapped_table(path)

    def test_two_batches_is_a_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "two.arrow"
        table = _table(4)
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table, max_chunksize=2)
        with pytest.raises(ValueError, match="2 record batches"):
            open_mapped_table(path)


class TestFixedWidthBytes:
    def test_views_each_row(self, tmp_path: Path) -> None:
        values = [bytes([row] * 4) for row in range(3)]
        path = tmp_path / "bytes.arrow"
        write_table_atomically(
            pa.table({"b": pa.array(values, type=pa.binary(4))}), path
        )

        view = fixed_width_bytes_of(open_mapped_table(path).column("b"), 4)

        assert view.shape == (3, 4)
        assert [row.tobytes() for row in view] == values

    def test_an_empty_column_is_an_empty_view(self) -> None:
        column = pa.chunked_array([], type=pa.binary(4))
        assert fixed_width_bytes_of(column, 4).shape == (0, 4)

    def test_a_sliced_chunk_views_only_its_rows(self) -> None:
        values = [bytes([row] * 2) for row in range(4)]
        sliced = pa.array(values, type=pa.binary(2)).slice(1, 2)
        view = fixed_width_bytes_of(pa.chunked_array([sliced]), 2)
        assert np.array_equal(view, np.array([[1, 1], [2, 2]], dtype=np.uint8))
