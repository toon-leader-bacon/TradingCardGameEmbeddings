"""Tests for turns_to_game_end_after_cast_metric.py's
TurnsToGameEndAfterCastMetric.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.turns_to_game_end_after_cast_metric import (  # noqa: E501
    TurnsToGameEndAfterCastMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_HEADER = [
    "num_turns",
    "user_turn_1_creatures_cast",
    "user_turn_1_non_creatures_cast",
    "user_turn_3_creatures_cast",
    "user_turn_3_non_creatures_cast",
    "oppo_turn_1_creatures_cast",
    "oppo_turn_1_non_creatures_cast",
]

_NA_ROW: dict = {column: float("nan") for column in _HEADER}


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


def _binder_with_alias(name: str, arena_id: str) -> CardBinder:
    binder = CardBinder()
    card = _make_card(name)
    binder.create(card)
    binder.register_alias(GameId.MTG, DataSource.ARENA, arena_id, card.nocab_uuid)
    return binder


def _uuid_for_arena_id(binder: CardBinder, arena_id: str) -> str:
    card = binder.get_by_alias(GameId.MTG, DataSource.ARENA, arena_id)
    assert card is not None
    return str(card.nocab_uuid)


def test_averages_num_turns_minus_first_cast_turn(tmp_path: Path) -> None:
    binder = _binder_with_alias("Owlbear", "1")
    metric = TurnsToGameEndAfterCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["num_turns"] = 10
    row["user_turn_3_creatures_cast"] = "1"

    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for_arena_id(binder, "1")
    assert df.loc[owlbear_uuid, "turns_to_game_end_after_cast"] == 7.0


def test_uses_first_occurrence_only_when_cast_on_multiple_turns(
    tmp_path: Path,
) -> None:
    binder = _binder_with_alias("Owlbear", "1")
    metric = TurnsToGameEndAfterCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["num_turns"] = 10
    row["user_turn_1_creatures_cast"] = "1"
    row["user_turn_3_creatures_cast"] = "1"

    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for_arena_id(binder, "1")
    # first cast turn is 1, not 3 - delta is 10 - 1 = 9, not 10 - 3 = 7.
    assert df.loc[owlbear_uuid, "turns_to_game_end_after_cast"] == 9.0


def test_counts_both_actors_occurrences(tmp_path: Path) -> None:
    binder = _binder_with_alias("Owlbear", "1")
    metric = TurnsToGameEndAfterCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["num_turns"] = 10
    row["oppo_turn_1_creatures_cast"] = "1"

    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for_arena_id(binder, "1")
    assert df.loc[owlbear_uuid, "turns_to_game_end_after_cast"] == 9.0


def test_card_never_cast_never_appears_in_output(tmp_path: Path) -> None:
    binder = _binder_with_alias("Owlbear", "1")
    metric = TurnsToGameEndAfterCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["num_turns"] = 10

    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.shape[0] == 0
