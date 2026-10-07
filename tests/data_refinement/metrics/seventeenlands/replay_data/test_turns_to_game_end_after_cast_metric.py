"""Tests for turns_to_game_end_after_cast_metric.py's
TurnsToGameEndAfterCastMetric."""

from pathlib import Path

from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    Actor,
    ReplayField,
)
from src.data_refinement.metrics.seventeenlands.replay_data.turns_to_game_end_after_cast_metric import (  # noqa: E501
    TurnsToGameEndAfterCastMetric,
)
from tests.data_refinement.metrics.seventeenlands.replay_data._replay_fixtures import (
    MORNINGSTAR,
    MORNINGSTAR_ID,
    OWLBEAR,
    OWLBEAR_ID,
    VERSION,
    binder_with_cards,
    field_column,
    row,
    scan_into_frame,
    uuid_for,
)

_LABEL = "turns_to_game_end_after_cast"


def _scan(tmp_path: Path, binder, rows):
    metric = TurnsToGameEndAfterCastMetric(VERSION, tmp_path / "out.parquet")
    return scan_into_frame(tmp_path, binder, rows, metric)


def test_one_entry_per_game_from_the_first_cast_turn_on_either_actor(
    tmp_path: Path,
) -> None:
    binder = binder_with_cards()
    cells = {
        field_column(Actor.OPPO, 2, ReplayField.CREATURES_CAST): OWLBEAR_ID,
        field_column(Actor.USER, 1, ReplayField.NON_CREATURES_CAST): OWLBEAR_ID,
        field_column(Actor.USER, 2, ReplayField.CREATURES_CAST): OWLBEAR_ID,
    }

    frame = _scan(tmp_path, binder, [row(cells, num_turns=9)])

    owlbear = str(uuid_for(binder, OWLBEAR))
    assert frame.loc[owlbear, "sample_count"] == 1
    assert frame.loc[owlbear, _LABEL] == 9 - 1


def test_averages_over_games(tmp_path: Path) -> None:
    binder = binder_with_cards()
    cast_turn_1 = {
        field_column(Actor.USER, 1, ReplayField.CREATURES_CAST): MORNINGSTAR_ID
    }
    cast_turn_2 = {
        field_column(Actor.OPPO, 2, ReplayField.CREATURES_CAST): MORNINGSTAR_ID
    }

    frame = _scan(
        tmp_path,
        binder,
        [row(cast_turn_1, num_turns=5), row(cast_turn_2, num_turns=10)],
    )

    morningstar = str(uuid_for(binder, MORNINGSTAR))
    assert frame.loc[morningstar, "sample_count"] == 2
    assert frame.loc[morningstar, _LABEL] == ((5 - 1) + (10 - 2)) / 2


def test_a_card_never_cast_is_absent(tmp_path: Path) -> None:
    frame = _scan(tmp_path, binder_with_cards(), [row(owlbear_deck=1)])

    assert frame.empty
