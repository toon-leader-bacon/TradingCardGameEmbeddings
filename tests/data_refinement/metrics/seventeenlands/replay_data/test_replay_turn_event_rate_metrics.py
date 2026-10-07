"""Tests for replay_turn_event_rate_metrics.py's two combat rates (the
ReplayTurnEventRateMetric base is exercised through them)."""

from pathlib import Path

import pytest

from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    Actor,
    ReplayField,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_turn_event_rate_metrics import (
    CombatDamagePushThroughRateMetric,
    CombatKillInvolvementRateMetric,
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


def _cell(actor: Actor, turn: int, field: ReplayField) -> str:
    return field_column(actor, turn, field)


_ATTACKED_U1 = _cell(Actor.USER, 1, ReplayField.CREATURES_ATTACKED)
_BLOCKING_U1 = _cell(Actor.USER, 1, ReplayField.CREATURES_BLOCKING)
_UNBLOCKED_U1 = _cell(Actor.USER, 1, ReplayField.CREATURES_UNBLOCKED)
_OPPO_KILLED_U1 = _cell(Actor.USER, 1, ReplayField.OPPO_CREATURES_KILLED_COMBAT)
_USER_KILLED_U1 = _cell(Actor.USER, 1, ReplayField.USER_CREATURES_KILLED_COMBAT)


class TestCombatKillInvolvementRateMetric:
    def _scan(self, tmp_path: Path, binder, rows):
        metric = CombatKillInvolvementRateMetric(VERSION, tmp_path / "out.parquet")
        return scan_into_frame(tmp_path, binder, rows, metric)

    @pytest.mark.parametrize("killed_column", [_OPPO_KILLED_U1, _USER_KILLED_U1])
    def test_either_sides_combat_kill_is_a_hit_for_every_fighter(
        self, tmp_path: Path, killed_column: str
    ) -> None:
        binder = binder_with_cards()
        cells = {
            _ATTACKED_U1: OWLBEAR_ID,
            _BLOCKING_U1: MORNINGSTAR_ID,
            killed_column: MORNINGSTAR_ID,
        }

        frame = self._scan(tmp_path, binder, [row(cells)])

        assert (
            frame.loc[str(uuid_for(binder, OWLBEAR)), "combat_kill_involvement_rate"]
            == 1.0
        )
        assert (
            frame.loc[
                str(uuid_for(binder, MORNINGSTAR)), "combat_kill_involvement_rate"
            ]
            == 1.0
        )

    def test_an_unmatched_kill_still_counts(self, tmp_path: Path) -> None:
        binder = binder_with_cards()
        cells = {_ATTACKED_U1: OWLBEAR_ID, _OPPO_KILLED_U1: UNKNOWN_ID}

        frame = self._scan(tmp_path, binder, [row(cells)])

        owlbear = str(uuid_for(binder, OWLBEAR))
        assert frame.loc[owlbear, "combat_kill_involvement_rate"] == 1.0

    def test_no_kill_is_no_hit(self, tmp_path: Path) -> None:
        binder = binder_with_cards()

        frame = self._scan(tmp_path, binder, [row({_ATTACKED_U1: OWLBEAR_ID})])

        owlbear = str(uuid_for(binder, OWLBEAR))
        assert frame.loc[owlbear, "combat_kill_involvement_rate"] == 0.0
        assert frame.loc[owlbear, "sample_count"] == 1

    def test_a_fighter_counts_once_per_half_turn(self, tmp_path: Path) -> None:
        binder = binder_with_cards()
        cells = {
            _ATTACKED_U1: f"{OWLBEAR_ID}|{OWLBEAR_ID}",
            _BLOCKING_U1: OWLBEAR_ID,
        }

        frame = self._scan(tmp_path, binder, [row(cells)])

        assert frame.loc[str(uuid_for(binder, OWLBEAR)), "sample_count"] == 1

    def test_half_turns_tally_across_actors_turns_and_games(
        self, tmp_path: Path
    ) -> None:
        binder = binder_with_cards()
        first = {
            _ATTACKED_U1: OWLBEAR_ID,
            _OPPO_KILLED_U1: MORNINGSTAR_ID,
            _cell(Actor.USER, 2, ReplayField.CREATURES_ATTACKED): OWLBEAR_ID,
        }
        second = {_cell(Actor.OPPO, 1, ReplayField.CREATURES_BLOCKING): OWLBEAR_ID}

        frame = self._scan(tmp_path, binder, [row(first), row(second)])

        owlbear = str(uuid_for(binder, OWLBEAR))
        assert frame.loc[owlbear, "sample_count"] == 3
        assert frame.loc[owlbear, "combat_kill_involvement_rate"] == pytest.approx(
            1 / 3
        )


class TestCombatDamagePushThroughRateMetric:
    def _scan(self, tmp_path: Path, binder, rows):
        metric = CombatDamagePushThroughRateMetric(VERSION, tmp_path / "out.parquet")
        return scan_into_frame(tmp_path, binder, rows, metric)

    def test_a_hit_is_per_card_not_turn_wide(self, tmp_path: Path) -> None:
        binder = binder_with_cards()
        cells = {
            _ATTACKED_U1: f"{OWLBEAR_ID}|{MORNINGSTAR_ID}",
            _UNBLOCKED_U1: OWLBEAR_ID,
        }

        frame = self._scan(tmp_path, binder, [row(cells)])

        rate = "combat_damage_push_through_rate"
        assert frame.loc[str(uuid_for(binder, OWLBEAR)), rate] == 1.0
        assert frame.loc[str(uuid_for(binder, MORNINGSTAR)), rate] == 0.0

    def test_every_attack_entry_counts_but_a_hit_counts_once(
        self, tmp_path: Path
    ) -> None:
        binder = binder_with_cards()
        cells = {
            _ATTACKED_U1: f"{OWLBEAR_ID}|{OWLBEAR_ID}",
            _UNBLOCKED_U1: f"{OWLBEAR_ID}|{OWLBEAR_ID}",
        }

        frame = self._scan(tmp_path, binder, [row(cells)])

        owlbear = str(uuid_for(binder, OWLBEAR))
        assert frame.loc[owlbear, "sample_count"] == 2
        assert frame.loc[owlbear, "combat_damage_push_through_rate"] == 0.5

    def test_unblocked_in_another_half_turn_is_no_hit(self, tmp_path: Path) -> None:
        binder = binder_with_cards()
        cells = {
            _ATTACKED_U1: OWLBEAR_ID,
            _cell(Actor.USER, 2, ReplayField.CREATURES_UNBLOCKED): OWLBEAR_ID,
        }

        frame = self._scan(tmp_path, binder, [row(cells)])

        owlbear = str(uuid_for(binder, OWLBEAR))
        assert frame.loc[owlbear, "combat_damage_push_through_rate"] == 0.0

    def test_a_card_that_only_blocks_is_absent(self, tmp_path: Path) -> None:
        binder = binder_with_cards()
        cells = {_BLOCKING_U1: OWLBEAR_ID, _ATTACKED_U1: MORNINGSTAR_ID}

        frame = self._scan(tmp_path, binder, [row(cells)])

        assert list(frame.index) == [str(uuid_for(binder, MORNINGSTAR))]

    def test_no_attacks_write_an_empty_table(self, tmp_path: Path) -> None:
        frame = self._scan(tmp_path, binder_with_cards(), [row()])

        assert frame.empty
