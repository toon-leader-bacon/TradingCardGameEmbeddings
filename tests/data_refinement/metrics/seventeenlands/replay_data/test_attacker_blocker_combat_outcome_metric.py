"""Tests for attacker_blocker_combat_outcome_metric.py's
AttackerBlockerCombatOutcomeMetric - the fan-out-over-turns streaming
shape (one row per qualifying half-turn, not per game).
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.attacker_blocker_combat_outcome_metric import (  # noqa: E501
    AttackerBlockerCombatOutcomeMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_HEADER = [
    "draft_id",
    "match_number",
    "game_number",
    "user_turn_1_creatures_attacked",
    "user_turn_1_creatures_blocking",
    "user_turn_1_user_creatures_killed_combat",
    "user_turn_1_oppo_creatures_killed_combat",
    "user_turn_2_creatures_attacked",
    "user_turn_2_creatures_blocking",
    "user_turn_2_user_creatures_killed_combat",
    "user_turn_2_oppo_creatures_killed_combat",
    "oppo_turn_1_creatures_attacked",
    "oppo_turn_1_creatures_blocking",
    "oppo_turn_1_user_creatures_killed_combat",
    "oppo_turn_1_oppo_creatures_killed_combat",
]

_NA_ROW: dict = {column: float("nan") for column in _HEADER}
_NA_ROW.update({"draft_id": "draft1", "match_number": 0, "game_number": 0})


def _make_card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder_with_owlbear(arena_id: str) -> CardBinder:
    binder = CardBinder()
    card = _make_card("Owlbear")
    binder.create(card)
    binder.register_alias(GameId.MTG, DataSource.ARENA, arena_id, card.nocab_uuid)
    return binder


def test_writes_one_row_per_half_turn_with_an_attacker(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    metric = AttackerBlockerCombatOutcomeMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["user_turn_1_creatures_attacked"] = "1"

    metric.accumulate(row)
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.num_rows == 1


def test_half_turn_with_no_attacker_writes_no_row(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    metric = AttackerBlockerCombatOutcomeMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["user_turn_1_creatures_blocking"] = "1"  # blocked, never attacked

    metric.accumulate(row)
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.num_rows == 0


def test_a_game_with_no_attacks_at_all_writes_zero_rows(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    metric = AttackerBlockerCombatOutcomeMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(dict(_NA_ROW))
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.num_rows == 0


def test_both_actors_half_turns_can_each_produce_a_row(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    metric = AttackerBlockerCombatOutcomeMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["user_turn_1_creatures_attacked"] = "1"
    row["oppo_turn_1_creatures_attacked"] = "1"

    metric.accumulate(row)
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.num_rows == 2
    actors = {r["actor"] for r in table.to_pylist()}
    assert actors == {"user", "oppo"}


def test_output_row_carries_actor_and_turn_identifiers(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    metric = AttackerBlockerCombatOutcomeMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["user_turn_2_creatures_attacked"] = "1"

    metric.accumulate(row)
    metric.finalize()

    output_row = pq.read_table(tmp_path / "out.parquet").to_pylist()[0]
    assert output_row["draft_id"] == "draft1"
    assert output_row["match_number"] == 0
    assert output_row["game_number"] == 0
    assert output_row["actor"] == "user"
    assert output_row["turn"] == 2


def test_net_kill_delta_is_attacker_favorable_positive(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    metric = AttackerBlockerCombatOutcomeMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["user_turn_1_creatures_attacked"] = "1"
    row["user_turn_1_oppo_creatures_killed_combat"] = "1"  # defender (oppo) lost one

    metric.accumulate(row)
    metric.finalize()

    output_row = pq.read_table(tmp_path / "out.parquet").to_pylist()[0]
    assert output_row["net_kill_delta"] == 1


def test_never_writes_a_deck_uuid_or_takes_a_deck_box() -> None:
    """This metric's identity is (draft_id, match_number, game_number,
    actor, turn), never a deck_uuid - see module docstring."""
    assert "deck_box" not in AttackerBlockerCombatOutcomeMetric.__init__.__annotations__
