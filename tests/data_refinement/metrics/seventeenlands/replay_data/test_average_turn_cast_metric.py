"""Tests for average_turn_cast_metric.py's AverageTurnCastMetric."""

from pathlib import Path

from src.data_refinement.metrics.seventeenlands.replay_data.average_turn_cast_metric import (
    AverageTurnCastMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    Actor,
    ReplayField,
)
from tests.data_refinement.metrics.seventeenlands.replay_data._replay_fixtures import (
    MORNINGSTAR,
    MORNINGSTAR_ID,
    OWLBEAR,
    OWLBEAR_ID,
    UNKNOWN_ID,
    VERSION,
    binder_with_cards,
    field_column,
    row,
    scan_into_frame,
    uuid_for,
)

_CREATURES = ReplayField.CREATURES_CAST
_NON_CREATURES = ReplayField.NON_CREATURES_CAST


def _scan(tmp_path: Path, binder, rows):
    metric = AverageTurnCastMetric(VERSION, tmp_path / "out.parquet")
    return scan_into_frame(tmp_path, binder, rows, metric)


def test_averages_every_cast_occurrence_over_both_actors_and_fields(
    tmp_path: Path,
) -> None:
    binder = binder_with_cards()
    cells = {
        field_column(Actor.USER, 1, _CREATURES): f"{OWLBEAR_ID}|{OWLBEAR_ID}",
        field_column(Actor.OPPO, 2, _NON_CREATURES): OWLBEAR_ID,
    }

    frame = _scan(tmp_path, binder, [row(cells)])

    owlbear = str(uuid_for(binder, OWLBEAR))
    assert frame.loc[owlbear, "sample_count"] == 3
    assert frame.loc[owlbear, "average_turn_cast"] == (1 + 1 + 2) / 3


def test_cards_tally_separately_across_games(tmp_path: Path) -> None:
    binder = binder_with_cards()
    first = {field_column(Actor.USER, 2, _CREATURES): MORNINGSTAR_ID}
    second = {field_column(Actor.USER, 1, _CREATURES): MORNINGSTAR_ID}

    frame = _scan(tmp_path, binder, [row(first), row(second)])

    assert frame.loc[str(uuid_for(binder, MORNINGSTAR)), "average_turn_cast"] == 1.5


def test_unmatched_and_uncast_cards_are_absent(tmp_path: Path) -> None:
    binder = binder_with_cards()
    cells = {field_column(Actor.USER, 1, _CREATURES): f"{UNKNOWN_ID}|{OWLBEAR_ID}"}

    frame = _scan(tmp_path, binder, [row(cells, morningstar_deck=1)])

    assert list(frame.index) == [str(uuid_for(binder, OWLBEAR))]
