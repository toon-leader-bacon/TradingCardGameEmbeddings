"""Tests for count_table.py: write_count_table and ratio_output."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data_refinement.metrics.seventeenlands.count_table import (
    ratio_output,
    write_count_table,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_length_association_metric import (
    GameLengthAssociationMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    read_version_metadata,
)
from src.schema.game_id import GameId

VERSION = MetricVersionMetadata(game=GameId.MTG, card_binder_version="v1")


def _tutor_counts(in_deck: list[float], tutored: list[float]) -> dict:
    return {
        "in_deck": np.array(in_deck, np.float64),
        "tutored": np.array(tutored, np.float64),
    }


class TestWriteCountTable:
    def test_writes_keys_counts_and_version(self, tmp_path: Path) -> None:
        path = write_count_table(
            tmp_path / "a" / "p.parquet",
            TutorTargetRateMetric,
            {"nocab_uuid": ["x", "y"]},
            _tutor_counts([2, 1], [1, 0]),
            VERSION,
        )

        table = pq.read_table(path)
        assert table.column_names == ["nocab_uuid", "in_deck", "tutored"]
        assert table.column("in_deck").to_pylist() == [2.0, 1.0]
        assert read_version_metadata(path) == VERSION

    def test_no_keys_still_writes_the_schema(self, tmp_path: Path) -> None:
        path = write_count_table(
            tmp_path / "p.parquet",
            TutorTargetRateMetric,
            {"nocab_uuid": []},
            _tutor_counts([], []),
            VERSION,
        )

        table = pq.read_table(path)
        assert table.num_rows == 0
        assert table.column_names == ["nocab_uuid", "in_deck", "tutored"]

    def test_baseline_row_has_null_keys(self, tmp_path: Path) -> None:
        path = write_count_table(
            tmp_path / "p.parquet",
            GameLengthAssociationMetric,
            {"nocab_uuid": ["x"]},
            {"games": np.array([1.0]), "value_sum": np.array([8.0])},
            VERSION,
            baseline=np.array([3.0, 30.0]),
        )

        rows = pq.read_table(path).to_pylist()
        assert rows[-1] == {"nocab_uuid": None, "games": 3.0, "value_sum": 30.0}

    def test_rejects_columns_the_metric_does_not_declare(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="count columns"):
            write_count_table(
                tmp_path / "p.parquet",
                TutorTargetRateMetric,
                {"nocab_uuid": ["x"]},
                {"tutored": np.array([1.0]), "in_deck": np.array([1.0])},
                VERSION,
            )

    def test_rejects_a_baseline_the_metric_does_not_have(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="HAS_BASELINE"):
            write_count_table(
                tmp_path / "p.parquet",
                TutorTargetRateMetric,
                {"nocab_uuid": ["x"]},
                _tutor_counts([1], [1]),
                VERSION,
                baseline=np.array([1.0, 1.0]),
            )

    def test_rejects_a_missing_baseline(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="HAS_BASELINE"):
            write_count_table(
                tmp_path / "p.parquet",
                GameLengthAssociationMetric,
                {"nocab_uuid": ["x"]},
                {"games": np.array([1.0]), "value_sum": np.array([8.0])},
                VERSION,
            )

    def test_rejects_columns_of_different_lengths(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="rows, expected"):
            write_count_table(
                tmp_path / "p.parquet",
                TutorTargetRateMetric,
                {"nocab_uuid": ["x", "y"]},
                _tutor_counts([1], [1]),
                VERSION,
            )


class TestRatioOutput:
    def test_divides_and_uses_the_denominator_as_sample_count(self) -> None:
        summed = pa.table({"k": ["a", "b"], "n": [1.0, 0.0], "d": [4.0, 2.0]})

        result = ratio_output(summed, ("k",), "n", "d", "rate")

        assert result.to_pylist() == [
            {"k": "a", "rate": 0.25, "sample_count": 4},
            {"k": "b", "rate": 0.0, "sample_count": 2},
        ]
        assert result.schema.field("sample_count").type == pa.int64()

    def test_drops_keys_with_a_zero_denominator(self) -> None:
        summed = pa.table({"k": ["a", "b"], "n": [0.0, 1.0], "d": [0.0, 1.0]})

        result = ratio_output(summed, ("k",), "n", "d", "rate")

        assert result.column("k").to_pylist() == ["b"]

    def test_empty_input_gives_an_empty_table_with_the_schema(self) -> None:
        summed = pa.table(
            {
                "k": pa.array([], pa.string()),
                "n": pa.array([], pa.float64()),
                "d": pa.array([], pa.float64()),
            }
        )

        result = ratio_output(summed, ("k",), "n", "d", "rate")

        assert result.num_rows == 0
        assert result.column_names == ["k", "rate", "sample_count"]
