"""Tests for tutor_target_pool_metric.py's TutorTargetPoolMetric, driven
through scan_game_csv as a real run drives it."""

from pathlib import Path

import pyarrow.parquet as pq

from src.data_refinement.metrics.seventeenlands.game_data.scanner import scan_game_csv
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_pool_metric import (
    TutorTargetPoolMetric,
)
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    VERSION,
    binder_with_cards,
    parser_for,
    row,
    uuid_for,
    write_csv,
)

_BINDER = binder_with_cards([OWLBEAR, MORNINGSTAR])
_OWLBEAR = str(uuid_for(_BINDER, OWLBEAR))
_MORNINGSTAR = str(uuid_for(_BINDER, MORNINGSTAR))


def _scan_rows(tmp_path: Path, rows: list[dict], block_size: int = 1 << 20) -> list:
    """Scan rows through a TutorTargetPoolMetric; its output rows."""
    metric = TutorTargetPoolMetric(VERSION, output_path=tmp_path / "out.parquet")
    csv_path = write_csv(tmp_path / "games.csv", rows)
    scan_game_csv(csv_path, [metric], parser_for(_BINDER), block_size=block_size)
    return pq.read_table(metric.finalize()).to_pylist()


def test_writes_one_row_per_pool_card_deck_union_sideboard(tmp_path: Path) -> None:
    rows = _scan_rows(
        tmp_path,
        [row(won=True, owlbear_sideboard=2, morningstar_deck=4, draft_id="d1")],
    )

    assert {r["pool_card_uuid"] for r in rows} == {_OWLBEAR, _MORNINGSTAR}
    assert all(
        (r["draft_id"], r["match_number"], r["game_number"]) == ("d1", 1, 1)
        for r in rows
    )


def test_tutored_flag_true_only_for_cards_present_in_tutored_columns(
    tmp_path: Path,
) -> None:
    rows = _scan_rows(
        tmp_path,
        [row(won=True, owlbear_deck=4, owlbear_tutored=1, morningstar_deck=4)],
    )

    by_uuid = {r["pool_card_uuid"]: r["tutored"] for r in rows}
    assert by_uuid == {_OWLBEAR: True, _MORNINGSTAR: False}


def test_empty_pool_writes_zero_rows(tmp_path: Path) -> None:
    assert _scan_rows(tmp_path, [row(won=True)]) == []


def test_card_in_both_deck_and_sideboard_appears_once_not_twice(
    tmp_path: Path,
) -> None:
    # Not a real game, but the pool union still dedupes defensively
    rows = _scan_rows(tmp_path, [row(won=True, owlbear_deck=4, owlbear_sideboard=2)])

    assert [r["pool_card_uuid"] for r in rows] == [_OWLBEAR]


def test_rows_fan_out_per_game_in_game_order(tmp_path: Path) -> None:
    rows = _scan_rows(
        tmp_path,
        [
            row(won=True, owlbear_deck=1, game_number=1),
            row(won=True, game_number=2),  # empty pool
            row(won=True, owlbear_deck=1, morningstar_deck=1, game_number=3),
        ],
        block_size=256,
    )

    assert [r["game_number"] for r in rows] == [1, 3, 3]


def test_never_hashes_pool_as_a_deck_uuid() -> None:
    """This metric's identity is (draft_id, match_number, game_number,
    pool_card_uuid), never a deck hash over the wider pool."""
    assert "deck_box" not in TutorTargetPoolMetric.__init__.__annotations__
