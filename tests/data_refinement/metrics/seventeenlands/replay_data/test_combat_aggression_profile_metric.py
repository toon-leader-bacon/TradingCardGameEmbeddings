"""Tests for combat_aggression_profile_metric.py's
CombatAggressionProfileMetric.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.replay_data.combat_aggression_profile_metric import (  # noqa: E501
    CombatAggressionProfileMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_HEADER = [
    "draft_id",
    "match_number",
    "game_number",
    "deck_Owlbear",
    "user_turn_1_creatures_attacked",
    "user_turn_2_creatures_attacked",
]


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


def _row(
    deck: int,
    turn_1_attacked: str | float = float("nan"),
    turn_2_attacked: str | float = float("nan"),
) -> dict:
    return {
        "draft_id": "draft1",
        "match_number": 0,
        "game_number": 0,
        "deck_Owlbear": deck,
        "user_turn_1_creatures_attacked": turn_1_attacked,
        "user_turn_2_creatures_attacked": turn_2_attacked,
    }


def test_writes_one_row_per_game_with_the_average_attackers_per_turn(
    tmp_path: Path,
) -> None:
    binder = _binder_with_owlbear("1")
    deck_box = DeckBox()
    metric = CombatAggressionProfileMetric(
        binder, _HEADER, GameId.MTG, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(deck=4, turn_1_attacked="1", turn_2_attacked="1"))
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["combat_aggression_profile"] == 1.0
    assert row["draft_id"] == "draft1"


def test_turns_with_no_attack_are_excluded_from_the_average(tmp_path: Path) -> None:
    """Averages only over user half-turns that had any attack at all,
    not over every scanned turn."""
    binder = _binder_with_owlbear("1")
    deck_box = DeckBox()
    metric = CombatAggressionProfileMetric(
        binder, _HEADER, GameId.MTG, deck_box, output_path=tmp_path / "out.parquet"
    )

    # Only turn 1 has an attack - turn 2 stays NaN (no attack) and must
    # not drag the average toward zero.
    metric.accumulate(_row(deck=4, turn_1_attacked="1"))
    metric.finalize()

    row = pq.read_table(tmp_path / "out.parquet").to_pylist()[0]
    assert row["combat_aggression_profile"] == 1.0


def test_never_attacking_writes_zero_profile_not_omitted(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    deck_box = DeckBox()
    metric = CombatAggressionProfileMetric(
        binder, _HEADER, GameId.MTG, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(deck=4))
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.num_rows == 1
    assert table.to_pylist()[0]["combat_aggression_profile"] == 0.0


def test_writes_the_deck_into_the_shared_deck_box(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    deck_box = DeckBox()
    metric = CombatAggressionProfileMetric(
        binder, _HEADER, GameId.MTG, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(deck=4, turn_1_attacked="1"))
    metric.finalize()

    assert len(list(deck_box.all_uuids(GameId.MTG))) == 1


def test_identical_decks_across_games_dedupe_in_the_deck_box(tmp_path: Path) -> None:
    binder = _binder_with_owlbear("1")
    deck_box = DeckBox()
    metric = CombatAggressionProfileMetric(
        binder, _HEADER, GameId.MTG, deck_box, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(deck=4, turn_1_attacked="1"))
    metric.accumulate(_row(deck=4, turn_1_attacked="1"))
    metric.finalize()

    assert len(list(deck_box.all_uuids(GameId.MTG))) == 1
