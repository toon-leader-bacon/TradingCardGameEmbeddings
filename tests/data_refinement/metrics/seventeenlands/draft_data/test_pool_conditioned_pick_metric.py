"""Tests for pool_conditioned_pick_metric.py's PoolConditionedPickMetric,
driven through scan_draft_csv."""

from pathlib import Path

from src.data_refinement.metrics.seventeenlands.draft_data.pool_conditioned_pick_metric import (
    PoolConditionedPickMetric,
)
from tests.data_refinement.metrics.seventeenlands.draft_data._draft_fixtures import (
    BOLT,
    MORNINGSTAR,
    OWLBEAR,
    VERSION,
    binder_with_cards,
    row,
    scan_into_frame,
    uuid_for,
)

_BINDER = binder_with_cards([OWLBEAR, MORNINGSTAR, BOLT])


def _metric(tmp_path: Path) -> PoolConditionedPickMetric:
    return PoolConditionedPickMetric(VERSION, tmp_path / "out.parquet")


def test_writes_pool_and_pack_options_and_pick(tmp_path: Path) -> None:
    rows = [row(OWLBEAR, owlbear=1, morningstar=1, bolt_pool=2, owlbear_pool=1)]

    frame = scan_into_frame(tmp_path, _BINDER, rows, _metric(tmp_path))

    (record,) = frame.to_dict("records")
    assert list(record["pool_uuids"]) == [
        str(uuid_for(_BINDER, OWLBEAR)),
        str(uuid_for(_BINDER, BOLT)),
    ]
    assert record["pick_uuid"] == str(uuid_for(_BINDER, OWLBEAR))
    assert list(frame.columns) == [
        "draft_id",
        "pack_number",
        "pick_number",
        "pool_uuids",
        "pack_option_uuids",
        "pick_uuid",
    ]


def test_a_card_with_a_zero_pool_count_is_absent(tmp_path: Path) -> None:
    rows = [row(OWLBEAR, owlbear=1, bolt_pool=0)]

    frame = scan_into_frame(tmp_path, _BINDER, rows, _metric(tmp_path))

    assert list(frame["pool_uuids"][0]) == []


def test_an_unmatched_pick_is_written_as_null(tmp_path: Path) -> None:
    frame = scan_into_frame(
        tmp_path, _BINDER, [row("Nonexistent Card", owlbear=1)], _metric(tmp_path)
    )

    assert frame["pick_uuid"].isna().all()
