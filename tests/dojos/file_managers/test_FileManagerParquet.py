from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.dojos.file_managers.FileManagerParquet import FileManagerParquet
from src.schema.splits import Split


def _write_source(path: Path, num_rows: int = 100) -> None:
    df = pd.DataFrame({
        "name": [f"card{i}" for i in range(num_rows)],
        "value": range(num_rows),
    })
    df.to_parquet(path, index=False)


def _read_all(reader) -> pd.DataFrame:
    chunks = list(reader)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


class TestFileManagerParquetInit:
    def test_raises_if_source_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FileManagerParquet(tmp_path / "missing.parquet", tmp_path / "out")

    def test_raises_if_not_parquet_suffix(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "source.csv"
        bad_file.write_text("name,value\na,1\n")

        with pytest.raises(ValueError):
            FileManagerParquet(bad_file, tmp_path / "out")

    def test_raises_if_source_empty(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "source.parquet"
        empty_file.touch()

        with pytest.raises(ValueError):
            FileManagerParquet(empty_file, tmp_path / "out")

    def test_raises_if_source_corrupt(self, tmp_path: Path) -> None:
        corrupt_file = tmp_path / "source.parquet"
        corrupt_file.write_text("not actually parquet")

        with pytest.raises(Exception):
            FileManagerParquet(corrupt_file, tmp_path / "out")

    def test_caches_source_schema(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)

        fm = FileManagerParquet(source, tmp_path / "out")

        assert set(fm._schema.names) == {"name", "value"}


class TestMakeSplits:
    def test_splits_rows_according_to_ratio(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=1000)

        fm = FileManagerParquet(source, tmp_path / "out", seed=0)
        readers = fm.make_splits(split_ratios=[8, 1, 1], batch_size=32)

        assert len(readers) == 3
        row_counts = [len(_read_all(reader)) for reader in readers]
        assert row_counts == [800, 100, 100]
        assert sum(row_counts) == 1000

    def test_no_rows_lost_or_duplicated(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=137)

        fm = FileManagerParquet(source, tmp_path / "out", seed=1)
        readers = fm.make_splits(
            split_ratios=[8, 1, 1], batch_size=16, load_row_group_batch_size=40)

        all_names = sorted(
            name
            for reader in readers
            for chunk in reader
            for name in chunk["name"]
        )
        expected_names = sorted(f"card{i}" for i in range(137))
        assert all_names == expected_names

    def test_single_split_gets_everything(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=25)

        fm = FileManagerParquet(source, tmp_path / "out", seed=0)
        readers = fm.make_splits(split_ratios=[1], batch_size=8)

        assert len(readers) == 1
        assert len(_read_all(readers[0])) == 25

    def test_delete_old_splits_removes_stale_files(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        out_dir = tmp_path / "out"

        fm = FileManagerParquet(source, out_dir, seed=0)
        fm.make_splits(split_ratios=[8, 1, 1], batch_size=8)
        stale_file = out_dir / "split__stale.parquet"
        stale_file.write_text("stale")

        fm.make_splits(split_ratios=[8, 1, 1], batch_size=8, delete_old_splits=True)

        assert not stale_file.exists()

    def test_keeps_old_splits_when_not_deleting(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        out_dir = tmp_path / "out"

        fm = FileManagerParquet(source, out_dir, seed=0)
        fm.make_splits(split_ratios=[8, 1, 1], batch_size=8)
        stale_file = out_dir / "split__stale.parquet"
        stale_file.write_text("stale")

        fm.make_splits(split_ratios=[8, 1, 1], batch_size=8, delete_old_splits=False)

        assert stale_file.exists()

    def test_preserves_nulls_in_nullable_int_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        df = pd.DataFrame({
            "name": [f"card{i}" for i in range(50)],
            "value": pd.array(
                [i if i % 7 != 0 else None for i in range(50)], dtype="Int64",
            ),
        })
        df.to_parquet(source, index=False)

        fm = FileManagerParquet(source, tmp_path / "out", seed=0)
        readers = fm.make_splits(
            split_ratios=[1], batch_size=8, load_row_group_batch_size=5)

        result = _read_all(readers[0])
        assert result["value"].isna().sum() == sum(1 for i in range(50) if i % 7 == 0)
        assert len(result) == 50

    def test_split_writers_all_close_even_if_one_batch_write_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=100)
        out_dir = tmp_path / "out"

        fm = FileManagerParquet(source, out_dir, seed=0)

        real_write_batch = fm._write_batch_to_splits
        call_count = {"n": 0}

        def _flaky_write_batch(batch, writers, splits, shuffle):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated failure mid-stream")
            return real_write_batch(batch, writers, splits, shuffle)

        monkeypatch.setattr(fm, "_write_batch_to_splits", _flaky_write_batch)

        with pytest.raises(RuntimeError, match="simulated failure mid-stream"):
            fm.make_splits(
                split_ratios=[8, 1, 1], batch_size=8, load_row_group_batch_size=10)

        for split_file in out_dir.glob(f"{fm.output_file_prefix}*.parquet"):
            # A writer left open by an unhandled exception never writes a
            # valid footer; ExitStack must have closed every writer
            # regardless of which one was mid-write when the error hit.
            pq.ParquetFile(split_file).schema_arrow


class TestShuffleSplitIndex:
    def test_reshuffles_and_preserves_row_count(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=200)

        fm = FileManagerParquet(source, tmp_path / "out", seed=3)
        fm.make_splits(split_ratios=[8, 1, 1], batch_size=32)

        reshuffled = fm.shuffle_split_index(0, batch_size=32)
        result = _read_all(reshuffled)

        assert len(result) == 160

    def test_row_content_unchanged_by_shuffle(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=50)

        fm = FileManagerParquet(source, tmp_path / "out", seed=5)
        readers = fm.make_splits(split_ratios=[1], batch_size=16)
        before = set(_read_all(readers[0])["name"])

        reshuffled = fm.shuffle_split_index(0, batch_size=16)
        after = set(_read_all(reshuffled)["name"])

        assert before == after

    def test_raises_if_split_file_missing(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)

        fm = FileManagerParquet(source, tmp_path / "out", seed=0)

        with pytest.raises(FileNotFoundError):
            fm.shuffle_split_index(0, batch_size=8)


class TestShuffleSplit:
    def test_dispatches_train_test_validation_by_index(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=100)

        fm = FileManagerParquet(source, tmp_path / "out", seed=0)
        fm.make_splits(split_ratios=[8, 1, 1], batch_size=16)

        train = _read_all(fm.shuffle_split(Split.TRAIN, batch_size=16))
        test = _read_all(fm.shuffle_split(Split.TEST, batch_size=16))
        validation = _read_all(fm.shuffle_split(Split.VALIDATION, batch_size=16))

        assert len(train) == 80
        assert len(test) == 10
        assert len(validation) == 10
