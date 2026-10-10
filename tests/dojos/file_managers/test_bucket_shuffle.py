from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.dojos.file_managers import bucket_shuffle
from src.dojos.file_managers.bucket_shuffle import (
    ShufflePlan,
    _rows_by_bucket,
    scatter_into_buckets,
)


def _source(path: Path, num_rows: int) -> pq.ParquetFile:
    pd.DataFrame({"value": range(num_rows)}).to_parquet(path, index=False)
    return pq.ParquetFile(path)


class TestShufflePlan:
    @pytest.mark.parametrize("fields", [(0, 10), (1, 0)])
    def test_fields_below_one_raise(self, fields: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            ShufflePlan(*fields)

    def test_a_small_source_is_one_bucket(self, tmp_path: Path) -> None:
        source_file = _source(tmp_path / "s.parquet", 100)
        assert ShufflePlan.for_source(source_file, 2**20, 10_000) == ShufflePlan(
            1, 10_000
        )

    def test_buckets_follow_the_in_memory_size(self, tmp_path: Path) -> None:
        source_file = _source(tmp_path / "s.parquet", 10_000)
        size = source_file.read().nbytes  # int64 values plus a validity bitmap

        plan = ShufflePlan.for_source(source_file, -(-size // 4), 10)

        assert plan.bucket_count == 4
        # Batches grow so each bucket averages 1,000 rows per write
        assert plan.scatter_batch_rows == plan.bucket_count * 1_000

    def test_sizes_by_decoded_rows_not_dictionary_encoded_bytes(
        self, tmp_path: Path
    ) -> None:
        # Long repeated strings: tiny once dictionary-encoded, big in memory
        path = tmp_path / "s.parquet"
        names = [["x" * 40 + str(i % 3)] * 10 for i in range(5_000)]
        pd.DataFrame({"names": names}).to_parquet(path, index=False)
        source_file = pq.ParquetFile(path)
        metadata = source_file.metadata
        encoded = sum(
            metadata.row_group(i).total_byte_size
            for i in range(metadata.num_row_groups)
        )
        in_memory = source_file.read().nbytes
        assert encoded * 10 < in_memory  # the case the estimate must catch

        plan = ShufflePlan.for_source(source_file, -(-in_memory // 8), 10)

        assert plan.bucket_count == 8

    def test_an_empty_source_is_one_bucket(self, tmp_path: Path) -> None:
        source_file = _source(tmp_path / "s.parquet", 0)
        assert ShufflePlan.for_source(source_file, 1, 10).bucket_count == 1

    def test_the_bucket_count_is_capped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bucket_shuffle, "_MAX_BUCKETS", 3)
        source_file = _source(tmp_path / "s.parquet", 10_000)
        assert ShufflePlan.for_source(source_file, 1, 10).bucket_count == 3

    @pytest.mark.parametrize("sizes", [(0, 10), (-1, 10), (100, 0)])
    def test_bad_sizes_raise(self, tmp_path: Path, sizes: tuple[int, int]) -> None:
        source_file = _source(tmp_path / "s.parquet", 10)
        with pytest.raises(ValueError):
            ShufflePlan.for_source(source_file, *sizes)


class TestScatterIntoBuckets:
    def test_every_row_lands_in_exactly_one_bucket(self, tmp_path: Path) -> None:
        source = tmp_path / "s.parquet"
        schema = _source(source, 1_000).schema_arrow
        buckets = tmp_path / "buckets"
        buckets.mkdir()

        paths = scatter_into_buckets(
            source, ShufflePlan(4, 64), schema, buckets, seed=0
        )

        assert len(paths) == 4 and all(path.exists() for path in paths)
        values = sorted(
            value
            for path in paths
            for value in pq.read_table(path)["value"].to_pylist()
        )
        assert values == list(range(1_000))
        # Uniform draws: no bucket far from a quarter of the rows
        assert all(150 < pq.read_metadata(path).num_rows < 350 for path in paths)

    def test_the_same_seed_gives_the_same_buckets(self, tmp_path: Path) -> None:
        source = tmp_path / "s.parquet"
        schema = _source(source, 300).schema_arrow

        def bucket_values(name: str) -> list[list[int]]:
            directory = tmp_path / name
            directory.mkdir()
            paths = scatter_into_buckets(
                source, ShufflePlan(3, 50), schema, directory, seed=7
            )
            return [pq.read_table(path)["value"].to_pylist() for path in paths]

        assert bucket_values("a") == bucket_values("b")


class TestRowsByBucket:
    def test_groups_rows_by_bucket_in_table_order(self) -> None:
        table = pa.table({"value": [10, 11, 12, 13, 14]})

        groups = list(_rows_by_bucket(table, np.array([2, 0, 2, 0, 2])))

        assert [(index, rows["value"].to_pylist()) for index, rows in groups] == [
            (0, [11, 13]),
            (2, [10, 12, 14]),
        ]
