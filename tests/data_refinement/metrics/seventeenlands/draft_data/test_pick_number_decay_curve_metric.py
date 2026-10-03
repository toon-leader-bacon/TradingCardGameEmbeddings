"""Tests for pick_number_decay_curve_metric.py's
PickNumberDecayCurveMetric, driven through scan_draft_csv."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.metrics.seventeenlands.draft_data.pick_number_decay_curve_metric import (
    PickNumberDecayCurveMetric,
)
from tests.data_refinement.metrics.seventeenlands.draft_data._draft_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    VERSION,
    binder_with_cards,
    row,
    scan_into_frame,
    uuid_for,
)

_BINDER = binder_with_cards([OWLBEAR, MORNINGSTAR])


def _metric(tmp_path: Path) -> PickNumberDecayCurveMetric:
    return PickNumberDecayCurveMetric(VERSION, tmp_path / "out.parquet")


def test_one_row_per_card_with_pick_number_indexed_vectors(tmp_path: Path) -> None:
    rows = [
        row(OWLBEAR, owlbear=1, pick_number=0),
        row(MORNINGSTAR, owlbear=1, morningstar=1, pick_number=1),
    ]

    frame = scan_into_frame(tmp_path, _BINDER, rows, _metric(tmp_path))

    owlbear = frame.set_index("nocab_uuid").loc[str(uuid_for(_BINDER, OWLBEAR))]
    assert list(owlbear["take_rate_by_pick_number"]) == [1.0, 0.0]
    assert list(owlbear["sample_count_by_pick_number"]) == [1, 1]


def test_an_unseen_bucket_has_a_null_rate_and_zero_count(tmp_path: Path) -> None:
    rows = [
        row(OWLBEAR, owlbear=1, pick_number=0),
        row(MORNINGSTAR, morningstar=1, pick_number=2),
    ]

    frame = scan_into_frame(tmp_path, _BINDER, rows, _metric(tmp_path))

    owlbear = frame.set_index("nocab_uuid").loc[str(uuid_for(_BINDER, OWLBEAR))]
    assert list(owlbear["sample_count_by_pick_number"]) == [1, 0, 0]
    assert pd.isna(list(owlbear["take_rate_by_pick_number"])[1:]).all()


def test_pick_numbers_past_max_bucket_count_fold_into_the_last_bucket(
    tmp_path: Path,
) -> None:
    last = PickNumberDecayCurveMetric.MAX_BUCKET_COUNT - 1
    rows = [row(OWLBEAR, owlbear=1, pick_number=last + 5)]

    frame = scan_into_frame(tmp_path, _BINDER, rows, _metric(tmp_path))

    (counts,) = frame["sample_count_by_pick_number"]
    assert len(counts) == last + 1
    assert counts[last] == 1


def test_partitions_hold_the_long_form_counts(tmp_path: Path) -> None:
    scan_into_frame(tmp_path, _BINDER, [row(OWLBEAR, owlbear=1)], _metric(tmp_path))

    assert pq.read_table(tmp_path / "out.parquet").column_names == [
        "nocab_uuid",
        "pick_number",
        "in_pack",
        "picked",
    ]


def test_no_rows_gives_an_empty_curve_table() -> None:
    summed = pa.table(
        {
            "nocab_uuid": pa.array([], pa.string()),
            "pick_number": pa.array([], pa.int64()),
            "in_pack": pa.array([], pa.float64()),
            "picked": pa.array([], pa.float64()),
        }
    )

    result = PickNumberDecayCurveMetric.output_from_counts(summed, None)

    assert result.num_rows == 0
    assert result.column_names == [
        "nocab_uuid",
        "take_rate_by_pick_number",
        "sample_count_by_pick_number",
    ]
