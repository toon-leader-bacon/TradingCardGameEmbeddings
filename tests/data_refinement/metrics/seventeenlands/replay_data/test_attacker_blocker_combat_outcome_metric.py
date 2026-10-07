"""Tests for attacker_blocker_combat_outcome_metric.py's
AttackerBlockerCombatOutcomeMetric."""

from pathlib import Path

from src.data_refinement.metrics.seventeenlands.replay_data.attacker_blocker_combat_outcome_metric import (  # noqa: E501
    AttackerBlockerCombatOutcomeMetric,
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


def _scan(tmp_path: Path, binder, rows: list[dict]):
    metric = AttackerBlockerCombatOutcomeMetric(VERSION, tmp_path / "out.parquet")
    return scan_into_frame(tmp_path, binder, rows, metric, index=None)


def _cell(actor: Actor, turn: int, field: ReplayField) -> str:
    return field_column(actor, turn, field)


def test_one_row_per_attacking_half_turn_with_its_groups(tmp_path: Path) -> None:
    binder = binder_with_cards()
    cells = {
        _cell(Actor.USER, 1, ReplayField.CREATURES_ATTACKED): (
            f"{MORNINGSTAR_ID}|{UNKNOWN_ID}|{OWLBEAR_ID}"
        ),
        _cell(Actor.USER, 1, ReplayField.CREATURES_BLOCKING): OWLBEAR_ID,
    }

    frame = _scan(tmp_path, binder, [row(cells)])

    owlbear, morningstar = (str(uuid_for(binder, n)) for n in (OWLBEAR, MORNINGSTAR))
    (output,) = frame.to_dict("records")
    assert output["actor"] == "user"
    assert output["turn"] == 1
    assert list(output["attacker_uuids"]) == [morningstar, owlbear]
    assert list(output["blocker_uuids"]) == [owlbear]
    assert output["net_kill_delta"] == 0


def test_half_turns_come_in_row_then_user_then_turn_order(tmp_path: Path) -> None:
    binder = binder_with_cards()
    first = {
        _cell(Actor.OPPO, 1, ReplayField.CREATURES_ATTACKED): OWLBEAR_ID,
        _cell(Actor.USER, 2, ReplayField.CREATURES_ATTACKED): OWLBEAR_ID,
        _cell(Actor.USER, 1, ReplayField.CREATURES_ATTACKED): OWLBEAR_ID,
    }
    second = {_cell(Actor.USER, 1, ReplayField.CREATURES_ATTACKED): OWLBEAR_ID}

    frame = _scan(
        tmp_path, binder, [row(first, game_number=1), row(second, game_number=2)]
    )

    assert list(zip(frame["game_number"], frame["actor"], frame["turn"])) == [
        (1, "user", 1),
        (1, "user", 2),
        (1, "oppo", 1),
        (2, "user", 1),
    ]
    assert frame["blocker_uuids"].map(list).tolist() == [[], [], [], []]


def test_net_kill_delta_is_defender_losses_minus_attacker_losses(
    tmp_path: Path,
) -> None:
    binder = binder_with_cards()
    user_attack = {
        _cell(Actor.USER, 1, ReplayField.CREATURES_ATTACKED): OWLBEAR_ID,
        _cell(Actor.USER, 1, ReplayField.OPPO_CREATURES_KILLED_COMBAT): (
            f"{OWLBEAR_ID}|{MORNINGSTAR_ID}"
        ),
        _cell(Actor.USER, 1, ReplayField.USER_CREATURES_KILLED_COMBAT): OWLBEAR_ID,
    }
    oppo_attack = {
        _cell(Actor.OPPO, 1, ReplayField.CREATURES_ATTACKED): OWLBEAR_ID,
        _cell(Actor.OPPO, 1, ReplayField.USER_CREATURES_KILLED_COMBAT): OWLBEAR_ID,
        _cell(Actor.OPPO, 1, ReplayField.OPPO_CREATURES_KILLED_COMBAT): (
            f"{OWLBEAR_ID}|{MORNINGSTAR_ID}|{UNKNOWN_ID}"
        ),
    }

    frame = _scan(tmp_path, binder, [row(user_attack), row(oppo_attack)])

    assert frame["net_kill_delta"].tolist() == [2 - 1, 1 - 2]


def test_a_half_turn_with_only_unmatched_attackers_is_skipped(
    tmp_path: Path,
) -> None:
    cells = {_cell(Actor.USER, 1, ReplayField.CREATURES_ATTACKED): UNKNOWN_ID}

    frame = _scan(tmp_path, binder_with_cards(), [row(cells)])

    assert frame.empty
