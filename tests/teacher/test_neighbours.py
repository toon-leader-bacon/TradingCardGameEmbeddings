from pathlib import Path

import numpy as np
import pytest

from src.teacher.neighbour_rules import CosineThresholdNeighbours, TopKNeighbours
from src.teacher import teacher_neighbours as teacher_neighbours_module
from src.teacher.teacher_neighbours import TeacherNeighbours, find_neighbours
from src.teacher.teacher_table import StaleTeacherError
from tests.teacher.teacher_fixtures import stamp, table_of

# Rows 0-2 point one way, 3 another, 4 opposite row 0
_VECTORS = [
    [1.0, 0.0],
    [0.9, 0.1],
    [0.8, 0.3],
    [0.0, 1.0],
    [-1.0, 0.0],
]


class TestTopKNeighboursRule:
    def test_keeps_the_first_k(self) -> None:
        assert TopKNeighbours(2).kept_count(np.array([0.9, 0.5, 0.1])) == 2

    def test_never_keeps_padding(self) -> None:
        assert TopKNeighbours(2).kept_count(np.array([0.9, -np.inf, -np.inf])) == 1

    def test_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError):
            TopKNeighbours(0)


class TestCosineThresholdNeighboursRule:
    def test_keeps_those_reaching_the_threshold(self) -> None:
        rule = CosineThresholdNeighbours(0.4, 8)

        assert rule.kept_count(np.array([0.9, 0.5, 0.1])) == 2

    def test_caps_at_max_neighbours(self) -> None:
        rule = CosineThresholdNeighbours(0.0, 1)

        assert rule.kept_count(np.array([0.9, 0.5, 0.1])) == 1

    def test_never_keeps_padding(self) -> None:
        rule = CosineThresholdNeighbours(-1.0, 8)

        assert rule.kept_count(np.array([0.2, -np.inf])) == 1

    @pytest.mark.parametrize(("min_cosine", "max_neighbours"), [(1.5, 4), (0.2, 0)])
    def test_rejects_invalid_options(
        self, min_cosine: float, max_neighbours: int
    ) -> None:
        with pytest.raises(ValueError):
            CosineThresholdNeighbours(min_cosine, max_neighbours)


class TestFindNeighbours:
    def test_top_k_ranks_by_cosine_and_skips_self(self) -> None:
        neighbours = find_neighbours(table_of(_VECTORS), TopKNeighbours(2))

        assert neighbours.neighbours_of(0).tolist() == [1, 2]
        assert neighbours.neighbours_of(4).tolist()[0] == 3
        cosines = neighbours.cosines[neighbours.offsets[0] : neighbours.offsets[1]]
        assert cosines[0] >= cosines[1]

    def test_threshold_gives_ragged_rows(self) -> None:
        neighbours = find_neighbours(
            table_of(_VECTORS), CosineThresholdNeighbours(0.9, 4)
        )

        assert neighbours.neighbours_of(0).tolist() == [1, 2]
        assert neighbours.neighbours_of(3).tolist() == []

    def test_unconfident_rows_take_no_part(self) -> None:
        table = table_of(_VECTORS, confident=[True, False, True, True, True])

        neighbours = find_neighbours(table, TopKNeighbours(2))

        assert neighbours.neighbours_of(1).tolist() == []
        assert 1 not in neighbours.indices.tolist()

    def test_k_beyond_the_candidates_keeps_what_exists(self) -> None:
        neighbours = find_neighbours(table_of(_VECTORS[:3]), TopKNeighbours(10))

        assert [len(neighbours.neighbours_of(row)) for row in range(3)] == [2, 2, 2]

    def test_carries_the_table_stamp(self) -> None:
        table = table_of(_VECTORS)

        assert find_neighbours(table, TopKNeighbours(1)).teacher_stamp == table.stamp


class TestTeacherNeighbours:
    def test_rejects_inconsistent_arrays(self) -> None:
        with pytest.raises(ValueError, match="offsets end"):
            TeacherNeighbours(
                teacher_stamp=stamp(),
                rule=TopKNeighbours(1),
                offsets=np.array([0, 2], dtype=np.int64),
                indices=np.array([1], dtype=np.int32),
                cosines=np.array([0.5], dtype=np.float32),
            )

    def test_neighbours_of_out_of_range_raises(self) -> None:
        neighbours = find_neighbours(table_of(_VECTORS), TopKNeighbours(1))

        with pytest.raises(IndexError):
            neighbours.neighbours_of(5)

    def test_save_open_round_trip(self, tmp_path: Path) -> None:
        rule = CosineThresholdNeighbours(0.5, 3)
        neighbours = find_neighbours(table_of(_VECTORS), rule)
        path = tmp_path / "n.neighbours.arrow"

        neighbours.save(path)
        reopened = TeacherNeighbours.open(path, rule)

        assert reopened.teacher_stamp == neighbours.teacher_stamp
        assert np.array_equal(reopened.offsets, neighbours.offsets)
        assert np.array_equal(reopened.indices, neighbours.indices)
        assert np.array_equal(reopened.cosines, neighbours.cosines)

    def test_open_for_another_rule_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "n.neighbours.arrow"
        find_neighbours(table_of(_VECTORS), TopKNeighbours(2)).save(path)

        with pytest.raises(StaleTeacherError):
            TeacherNeighbours.open(path, TopKNeighbours(3))


class TestFindNeighboursBlocks:
    def test_one_row_blocks_match_one_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table = table_of(_VECTORS)
        whole = find_neighbours(table, TopKNeighbours(3))

        monkeypatch.setattr(teacher_neighbours_module, "_ROWS_PER_BLOCK", 1)
        blocked = find_neighbours(table, TopKNeighbours(3))

        assert np.array_equal(whole.offsets, blocked.offsets)
        assert np.array_equal(whole.indices, blocked.indices)
        assert np.array_equal(whole.cosines, blocked.cosines)

    def test_a_single_confident_row_has_no_neighbours(self) -> None:
        table = table_of(_VECTORS, confident=[False, False, True, False, False])

        neighbours = find_neighbours(table, TopKNeighbours(3))

        assert neighbours.indices.tolist() == []
        assert neighbours.offsets.tolist() == [0] * 6
