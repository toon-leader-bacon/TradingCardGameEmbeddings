"""Tests for combat_aggression_profile_metric.py's
CombatAggressionProfileMetric."""

from pathlib import Path

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.replay_data.combat_aggression_profile_metric import (  # noqa: E501
    CombatAggressionProfileMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    Actor,
    ReplayField,
)
from src.schema.game_id import GameId
from tests.data_refinement.metrics.seventeenlands.replay_data._replay_fixtures import (
    MORNINGSTAR_ID,
    OWLBEAR_ID,
    UNKNOWN_ID,
    VERSION,
    binder_with_cards,
    field_column,
    row,
    scan_into_frame,
)


def _attacked(actor: Actor, turn: int) -> str:
    return field_column(actor, turn, ReplayField.CREATURES_ATTACKED)


def _scan(tmp_path: Path, rows: list[dict], deck_box: DeckBox | None = None):
    metric = CombatAggressionProfileMetric(
        VERSION, deck_box or DeckBox(), tmp_path / "out.parquet"
    )
    return scan_into_frame(tmp_path, binder_with_cards(), rows, metric, index=None)


def test_one_row_per_game_averaging_matched_attackers_over_attacking_turns(
    tmp_path: Path,
) -> None:
    cells = {
        _attacked(Actor.USER, 1): f"{OWLBEAR_ID}|{MORNINGSTAR_ID}|{UNKNOWN_ID}",
        _attacked(Actor.USER, 2): OWLBEAR_ID,
    }

    frame = _scan(tmp_path, [row(cells, owlbear_deck=1), row(owlbear_deck=1)])

    assert frame["combat_aggression_profile"].tolist() == [pytest.approx(1.5), 0.0]
    assert frame["draft_id"].tolist() == ["draft1", "draft1"]


def test_oppo_attacks_and_unmatched_only_turns_are_ignored(tmp_path: Path) -> None:
    cells = {
        _attacked(Actor.USER, 1): OWLBEAR_ID,
        _attacked(Actor.USER, 2): UNKNOWN_ID,
        _attacked(Actor.OPPO, 1): f"{OWLBEAR_ID}|{OWLBEAR_ID}",
    }

    frame = _scan(tmp_path, [row(cells, owlbear_deck=1)])

    assert frame["combat_aggression_profile"].tolist() == [1.0]


def test_each_games_deck_lands_in_the_deck_box_once(tmp_path: Path) -> None:
    deck_box = DeckBox()
    rows = [
        row(owlbear_deck=1, game_number=1),
        row(owlbear_deck=1, game_number=2),
        row(morningstar_deck=1, game_number=3),
    ]

    frame = _scan(tmp_path, rows, deck_box)

    assert len(list(deck_box.all_uuids(GameId.MTG))) == 2
    assert frame["deck_uuid"].nunique() == 2
    assert set(frame["deck_uuid"]) == {
        str(uuid) for uuid in deck_box.all_uuids(GameId.MTG)
    }
