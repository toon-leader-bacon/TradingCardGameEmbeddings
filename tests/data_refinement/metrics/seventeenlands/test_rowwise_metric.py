"""Tests for rowwise_metric.py's RowwiseMetric adapter."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.metrics.seventeenlands.rowwise_metric import (
    RowFailures,
    RowwiseMetric,
)


class _RowRecorder:
    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.rows: list[dict] = []
        self._fail_on = fail_on or set()

    def accumulate(self, row: dict) -> None:
        if row["id"] in self._fail_on:
            raise KeyError(f"bad row {row['id']}")
        self.rows.append(row)

    def finalize(self) -> Path:
        return Path("inner.parquet")


def _frame_of(chunk: pd.DataFrame) -> pd.DataFrame:
    return chunk


def test_feeds_every_row_in_order_as_a_dict() -> None:
    inner = _RowRecorder()
    metric: RowwiseMetric[pd.DataFrame] = RowwiseMetric(inner, _frame_of)

    metric.accumulate(pd.DataFrame({"id": [1, 2, 3], "won": [True, False, True]}))

    assert [row["id"] for row in inner.rows] == [1, 2, 3]
    assert inner.rows[1] == {"id": 2, "won": False}


def test_bad_rows_are_skipped_then_summarized_once() -> None:
    inner = _RowRecorder(fail_on={2, 4})
    metric: RowwiseMetric[pd.DataFrame] = RowwiseMetric(inner, _frame_of)

    with pytest.raises(RowFailures, match="2 of 5 rows raised") as raised:
        metric.accumulate(pd.DataFrame({"id": [1, 2, 3, 4, 5]}))

    assert [row["id"] for row in inner.rows] == [1, 3, 5]
    assert isinstance(raised.value.__cause__, KeyError)
    assert "bad row 2" in str(raised.value.__cause__)


def test_finalize_delegates_to_the_inner_metric() -> None:
    metric: RowwiseMetric[pd.DataFrame] = RowwiseMetric(_RowRecorder(), _frame_of)

    assert metric.finalize() == Path("inner.parquet")


def test_repr_names_the_inner_metric() -> None:
    inner = _RowRecorder()

    assert repr(inner) in repr(RowwiseMetric(inner, _frame_of))
