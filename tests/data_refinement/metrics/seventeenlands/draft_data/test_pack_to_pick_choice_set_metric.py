"""Tests for draft_choice_stream_metric.py's DraftChoiceStreamMetric,
through PackToPickChoiceSetMetric, driven through scan_draft_csv."""

from pathlib import Path

from src.data_refinement.metrics.seventeenlands.draft_data.pack_to_pick_choice_set_metric import (
    PackToPickChoiceSetMetric,
)
from tests.data_refinement.metrics.seventeenlands.draft_data._draft_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    PICK_TWO_HEADER,
    VERSION,
    binder_with_cards,
    row,
    scan_into_frame,
    uuid_for,
)

_BINDER = binder_with_cards([OWLBEAR, MORNINGSTAR])
_OWLBEAR = str(uuid_for(_BINDER, OWLBEAR))
_MORNINGSTAR = str(uuid_for(_BINDER, MORNINGSTAR))


def _metric(tmp_path: Path) -> PackToPickChoiceSetMetric:
    return PackToPickChoiceSetMetric(VERSION, tmp_path / "out" / "p.parquet")


def test_writes_one_row_with_pack_options_and_pick(tmp_path: Path) -> None:
    rows = [row(MORNINGSTAR, owlbear=1, morningstar=1, pack_number=1, pick_number=2)]

    frame = scan_into_frame(tmp_path, _BINDER, rows, _metric(tmp_path))

    (record,) = frame.to_dict("records")
    assert record["draft_id"] == "draft1"
    assert (record["pack_number"], record["pick_number"]) == (1, 2)
    assert list(record["pack_option_uuids"]) == [_OWLBEAR, _MORNINGSTAR]
    assert record["pick_uuid"] == _MORNINGSTAR
    assert list(frame.columns) == [
        "draft_id",
        "pack_number",
        "pick_number",
        "pack_option_uuids",
        "pick_uuid",
    ]


def test_an_unmatched_pick_is_written_as_null(tmp_path: Path) -> None:
    frame = scan_into_frame(
        tmp_path, _BINDER, [row("Nonexistent Card", owlbear=1)], _metric(tmp_path)
    )

    assert frame["pick_uuid"].isna().all()


def test_one_row_per_input_row_in_order(tmp_path: Path) -> None:
    rows = [row(OWLBEAR, owlbear=1, pick_number=i) for i in range(5)]

    frame = scan_into_frame(tmp_path, _BINDER, rows, _metric(tmp_path))

    assert frame["pick_number"].tolist() == [0, 1, 2, 3, 4]


def test_a_row_with_nothing_present_has_an_empty_option_list(tmp_path: Path) -> None:
    frame = scan_into_frame(tmp_path, _BINDER, [row(OWLBEAR)], _metric(tmp_path))

    assert list(frame["pack_option_uuids"][0]) == []


def test_pick_two_rows_are_skipped(tmp_path: Path) -> None:
    rows = [
        row(OWLBEAR, owlbear=1, morningstar=1, pick_2=MORNINGSTAR),
        row(OWLBEAR, owlbear=1, pick_number=1),
        row(OWLBEAR, owlbear=1, pick_number=2, pick_2="Unmatched Card"),
    ]

    frame = scan_into_frame(tmp_path, _BINDER, rows, _metric(tmp_path), PICK_TWO_HEADER)

    assert frame["pick_number"].tolist() == [1]


def test_finalize_is_idempotent(tmp_path: Path) -> None:
    metric = _metric(tmp_path)
    scan_into_frame(tmp_path, _BINDER, [row(OWLBEAR, owlbear=1)], metric)

    assert metric.finalize() == tmp_path / "out" / "p.parquet"
