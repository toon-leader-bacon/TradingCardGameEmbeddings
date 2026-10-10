from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.dojos.file_managers.file_manager_parquet import FileManagerParquet
from src.schema.splits import Split


def _write_source(path: Path, num_rows: int = 100) -> None:
    df = pd.DataFrame(
        {
            "name": [f"card{i}" for i in range(num_rows)],
            "value": range(num_rows),
        }
    )
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
            split_ratios=[8, 1, 1], batch_size=16, load_row_group_batch_size=40
        )

        all_names = sorted(
            name for reader in readers for chunk in reader for name in chunk["name"]
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
        # A fourth split left by an older 4-way split
        stale_file = out_dir / "split__split_3.parquet"
        stale_file.write_text("stale")

        fm.make_splits(split_ratios=[8, 1, 1], batch_size=8, delete_old_splits=True)

        assert not stale_file.exists()

    def test_keeps_old_splits_when_not_deleting(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        out_dir = tmp_path / "out"

        fm = FileManagerParquet(source, out_dir, seed=0)
        fm.make_splits(split_ratios=[8, 1, 1], batch_size=8)
        stale_file = out_dir / "split__split_3.parquet"
        stale_file.write_text("stale")

        fm.make_splits(split_ratios=[8, 1, 1], batch_size=8, delete_old_splits=False)

        assert stale_file.exists()

    def test_preserves_nulls_in_nullable_int_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        df = pd.DataFrame(
            {
                "name": [f"card{i}" for i in range(50)],
                "value": pd.array(
                    [i if i % 7 != 0 else None for i in range(50)],
                    dtype="Int64",
                ),
            }
        )
        df.to_parquet(source, index=False)

        fm = FileManagerParquet(source, tmp_path / "out", seed=0)
        readers = fm.make_splits(
            split_ratios=[1], batch_size=8, load_row_group_batch_size=5
        )

        result = _read_all(readers[0])
        assert result["value"].isna().sum() == sum(1 for i in range(50) if i % 7 == 0)
        assert len(result) == 50

    def test_a_failure_mid_write_keeps_the_old_splits_and_no_temporaries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=100)
        out_dir = tmp_path / "out"
        fm = FileManagerParquet(source, out_dir, seed=0)
        fm.make_splits(split_ratios=[8, 1, 1])
        old_train = _read_all(fm.reader_for(Split.TRAIN))

        real_write_bucket = fm._write_bucket_to_splits
        call_count = {"n": 0}

        def _flaky_write_bucket(bucket, writers, splits):  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated failure mid-stream")
            return real_write_bucket(bucket, writers, splits)

        monkeypatch.setattr(fm, "_write_bucket_to_splits", _flaky_write_bucket)

        # Tiny buckets: several pass-2 writes, the second one fails
        with pytest.raises(RuntimeError, match="simulated failure mid-stream"):
            fm.make_splits(split_ratios=[8, 1, 1], bucket_bytes=100)

        # The old, complete splits are still the ones on disk
        assert fm.splits_exist()
        pd.testing.assert_frame_equal(_read_all(fm.reader_for(Split.TRAIN)), old_train)
        assert not list((out_dir / ".shuffle_tmp").rglob("*"))


class TestSplitPath:
    def test_names_each_splits_file_under_the_output_directory(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        fm = FileManagerParquet(source, tmp_path / "out", output_file_prefix="x")
        fm.make_splits()

        for split in (Split.TRAIN, Split.TEST, Split.VALIDATION):
            path = fm.split_path(split)
            assert path.parent == tmp_path / "out"
            assert path.exists()
        assert len({fm.split_path(split) for split in Split}) == 3


class TestSplitsExist:
    def test_false_before_make_splits(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        fm = FileManagerParquet(source, tmp_path / "out", seed=0)

        assert fm.splits_exist() is False

    def test_true_after_make_splits(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        fm = FileManagerParquet(source, tmp_path / "out", seed=0)
        fm.make_splits(split_ratios=[8, 1, 1], batch_size=8)

        assert fm.splits_exist() is True

    def test_false_if_only_some_split_files_present(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)
        out_dir = tmp_path / "out"
        fm = FileManagerParquet(source, out_dir, seed=0)
        fm.make_splits(split_ratios=[8, 1, 1], batch_size=8)
        (out_dir / f"{fm.output_file_prefix}_validation.parquet").unlink()

        assert fm.splits_exist() is False


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


def _write_grouped_source(path: Path, num_groups: int, rows_per_group: int) -> None:
    df = pd.DataFrame(
        {
            "deck_uuid": [
                f"deck{group}"
                for group in range(num_groups)
                for _ in range(rows_per_group)
            ],
            "value": range(num_groups * rows_per_group),
        }
    )
    df.to_parquet(path, index=False)


class TestGroupSplits:
    def test_every_group_lands_in_exactly_one_split(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_grouped_source(source, num_groups=200, rows_per_group=5)

        fm = FileManagerParquet(
            source, tmp_path / "out", seed=3, split_group_column="deck_uuid"
        )
        # Small streaming batches, so one group spans several batches.
        readers = fm.make_splits(
            split_ratios=[8, 1, 1], batch_size=16, load_row_group_batch_size=7
        )

        groups_per_split = [set(_read_all(reader)["deck_uuid"]) for reader in readers]
        assert groups_per_split[0].isdisjoint(groups_per_split[1])
        assert groups_per_split[0].isdisjoint(groups_per_split[2])
        assert groups_per_split[1].isdisjoint(groups_per_split[2])
        assert sum(len(groups) for groups in groups_per_split) == 200

    def test_no_rows_lost_and_ratios_roughly_kept(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_grouped_source(source, num_groups=2000, rows_per_group=2)

        fm = FileManagerParquet(
            source, tmp_path / "out", seed=0, split_group_column="deck_uuid"
        )
        readers = fm.make_splits(split_ratios=[8, 1, 1])

        row_counts = [len(_read_all(reader)) for reader in readers]
        assert sum(row_counts) == 4000
        assert 2900 < row_counts[0] < 3500
        assert 250 < row_counts[1] < 550
        assert 250 < row_counts[2] < 550

    def test_same_seed_gives_same_assignment(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_grouped_source(source, num_groups=100, rows_per_group=3)

        def train_groups(output_name: str) -> set[str]:
            fm = FileManagerParquet(
                source, tmp_path / output_name, seed=11, split_group_column="deck_uuid"
            )
            readers = fm.make_splits(split_ratios=[8, 1, 1])
            return set(_read_all(readers[0])["deck_uuid"])

        assert train_groups("first") == train_groups("second")

    def test_raises_on_unknown_group_column(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=10)

        with pytest.raises(ValueError, match="split_group_column"):
            FileManagerParquet(source, tmp_path / "out", split_group_column="nope")


def _write_sorted_source(path: Path, num_sets: int, rows_per_set: int) -> None:
    """Rows sorted by set, like the 17lands metric files."""
    df = pd.DataFrame(
        {
            "set": [f"set{s}" for s in range(num_sets) for _ in range(rows_per_set)],
            "value": range(num_sets * rows_per_set),
        }
    )
    df.to_parquet(path, index=False)


class TestUniformShuffle:
    @pytest.mark.parametrize("bucket_bytes", [256 * 2**20, 2_000])
    def test_a_set_sorted_source_is_mixed_from_the_first_rows(
        self, tmp_path: Path, bucket_bytes: int
    ) -> None:
        # One bucket (whole-file shuffle) and many buckets (two passes)
        source = tmp_path / "source.parquet"
        _write_sorted_source(source, num_sets=10, rows_per_set=1_000)
        fm = FileManagerParquet(source, tmp_path / "out", seed=0)

        train = _read_all(fm.make_splits([8, 1, 1], bucket_bytes=bucket_bytes)[0])

        # Within-10k-batch shuffling would give only set0 here
        assert train["set"].iloc[:200].nunique() == 10

    def test_many_buckets_lose_and_duplicate_no_row(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_sorted_source(source, num_sets=5, rows_per_set=400)
        fm = FileManagerParquet(source, tmp_path / "out", seed=2)

        readers = fm.make_splits([8, 1, 1], bucket_bytes=1_000)

        values = sorted(v for reader in readers for v in _read_all(reader)["value"])
        assert values == list(range(2_000))

    def test_the_same_seed_gives_the_same_order(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_sorted_source(source, num_sets=4, rows_per_set=500)

        def train_values(output_name: str) -> list[int]:
            fm = FileManagerParquet(source, tmp_path / output_name, seed=9)
            reader = fm.make_splits([8, 1, 1], bucket_bytes=2_000)[0]
            return list(_read_all(reader)["value"])

        assert train_values("first") == train_values("second")

    def test_split_files_keep_small_row_groups(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=20_000)
        fm = FileManagerParquet(source, tmp_path / "out", seed=0)

        fm.make_splits([1])

        metadata = pq.ParquetFile(fm.split_path(Split.TRAIN)).metadata
        assert metadata.num_row_groups == 3
        assert metadata.row_group(0).num_rows == 8_000

    def test_temporaries_are_removed_after_a_clean_run(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=500)
        out_dir = tmp_path / "out"
        fm = FileManagerParquet(source, out_dir, seed=0)

        fm.make_splits([8, 1, 1], bucket_bytes=500)

        assert not (out_dir / ".shuffle_tmp").exists()
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "split__test.parquet",
            "split__train.parquet",
            "split__validation.parquet",
        ]

    def test_a_killed_runs_temporaries_are_swept_but_not_other_prefixes(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=50)
        temp = tmp_path / "out" / ".shuffle_tmp"
        (temp / "a_buckets_x1y2").mkdir(parents=True)
        (temp / "a_buckets_x1y2" / "bucket_00000.parquet").write_text("stale")
        (temp / "a_train.parquet.partial").write_text("stale")
        (temp / "a_b_train.parquet.partial").write_text("another dojo")

        FileManagerParquet(source, tmp_path / "out", "a", seed=0).make_splits([8, 1, 1])

        assert [p.name for p in temp.iterdir()] == ["a_b_train.parquet.partial"]

    def test_re_splitting_a_prefix_keeps_a_longer_prefix_splits(
        self, tmp_path: Path
    ) -> None:
        # e.g. sts2_runs.card_win_rate vs sts2_runs.card_win_rate_at_act2
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=50)
        out_dir = tmp_path / "out"
        longer = FileManagerParquet(source, out_dir, "win_rate_at_act2", seed=0)
        longer.make_splits([8, 1, 1])

        FileManagerParquet(source, out_dir, "win_rate", seed=0).make_splits([8, 1, 1])

        assert longer.splits_exist()


class TestSplitOrderStamp:
    def test_every_split_file_is_stamped(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=30)
        fm = FileManagerParquet(source, tmp_path / "out", seed=0)

        fm.make_splits([8, 1, 1])

        for split in Split:
            metadata = pq.read_schema(fm.split_path(split)).metadata
            assert metadata[b"nocab_split_order"] == b"uniform_shuffle"

    def test_unstamped_older_splits_count_as_missing(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=30)
        fm = FileManagerParquet(source, tmp_path / "out", seed=0)
        (tmp_path / "out").mkdir()
        for split in Split:
            _write_source(fm.split_path(split), num_rows=10)

        assert fm.splits_exist() is False

    def test_the_source_metadata_is_kept_beside_the_stamp(self, tmp_path: Path) -> None:
        source = tmp_path / "source.parquet"
        _write_source(source, num_rows=30)
        fm = FileManagerParquet(source, tmp_path / "out", seed=0)

        fm.make_splits([8, 1, 1])

        split_metadata = pq.read_schema(fm.split_path(Split.TRAIN)).metadata
        assert set(fm.schema.metadata or {}) < set(split_metadata)
