from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from src.data_refinement.snapshots.arrow_files import (
    open_mapped_table,
    write_table_atomically,
)
from src.teacher.teacher_table import StaleTeacherError, TeacherStamp, TeacherTable
from tests.teacher.teacher_fixtures import stamp, table_of


class TestTeacherStamp:
    def test_round_trips_through_schema_metadata(self) -> None:
        original = stamp()

        assert (
            TeacherStamp.from_schema_metadata(original.to_schema_metadata()) == original
        )

    def test_tag_is_stable_and_tells_stamps_apart(self) -> None:
        assert stamp().tag() == stamp().tag()
        assert stamp().tag() != stamp(fit_digest="other").tag()
        assert stamp().tag() != replace(stamp(), card_tier_key="other").tag()

    def test_missing_metadata_raises_stale_teacher(self) -> None:
        with pytest.raises(StaleTeacherError):
            TeacherStamp.from_schema_metadata(None)

    def test_missing_own_entry_raises_stale_teacher(self) -> None:
        metadata = stamp().deck_snapshot.to_schema_metadata()

        with pytest.raises(StaleTeacherError):
            TeacherStamp.from_schema_metadata(metadata)


class TestTeacherTable:
    def test_save_open_round_trip(self, tmp_path: Path) -> None:
        table = table_of([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [True, False, True])
        path = tmp_path / "t.teacher.arrow"

        table.save(path)
        reopened = TeacherTable.open(path)

        assert reopened.source_game == table.source_game
        assert reopened.card_uuids == table.card_uuids
        assert np.array_equal(reopened.vectors, table.vectors)
        assert reopened.vectors.dtype == np.float32
        assert np.array_equal(reopened.deck_counts, table.deck_counts)
        assert reopened.confident.tolist() == [True, False, True]
        assert reopened.stamp == table.stamp
        assert reopened.dim == 2

    def test_other_format_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "t.teacher.arrow"
        table_of([[1.0], [2.0]]).save(path)
        saved = open_mapped_table(path)
        metadata = dict(saved.schema.metadata)
        metadata[b"teacher_table_format_version"] = b"999"
        write_table_atomically(saved.replace_schema_metadata(metadata), path)

        with pytest.raises(StaleTeacherError):
            TeacherTable.open(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TeacherTable.open(tmp_path / "absent.arrow")

    def test_rejects_misshapen_arrays(self) -> None:
        table = table_of([[1.0], [2.0]])

        with pytest.raises(ValueError):
            replace(table, confident=np.array([True]))

    def test_not_a_teacher_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "other.arrow"
        write_table_atomically(pa.table({"x": [1]}), path)

        with pytest.raises(StaleTeacherError):
            TeacherTable.open(path)
