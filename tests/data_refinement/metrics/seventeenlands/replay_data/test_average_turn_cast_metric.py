"""Tests for average_turn_cast_metric.py's AverageTurnCastMetric."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.average_turn_cast_metric import (
    AverageTurnCastMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_HEADER = [
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


def test_averages_turn_number_across_every_occurrence(tmp_path: Path) -> None:
    binder = _binder_with_alias("Owlbear", "1")
    metric = AverageTurnCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["user_turn_1_creatures_cast"] = "1"
    row["user_turn_3_creatures_cast"] = "1"

    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for_arena_id(binder, "1")
    assert df.loc[owlbear_uuid, "average_turn_cast"] == 2.0


def test_counts_both_actors_occurrences(tmp_path: Path) -> None:
    """No deck-membership conditioning - a card cast by either side
    counts."""
    binder = _binder_with_alias("Owlbear", "1")
    metric = AverageTurnCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["oppo_turn_1_creatures_cast"] = "1"

    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for_arena_id(binder, "1")
    assert df.loc[owlbear_uuid, "average_turn_cast"] == 1.0


def test_creature_and_non_creature_cast_columns_both_count(tmp_path: Path) -> None:
    binder = _binder_with_alias("Owlbear", "1")
    metric = AverageTurnCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["user_turn_1_non_creatures_cast"] = "1"

    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for_arena_id(binder, "1")
    assert df.loc[owlbear_uuid, "sample_count"] == 1


def test_multiple_occurrences_of_the_same_card_in_one_game_all_tally(
    tmp_path: Path,
) -> None:
    """A card with 2 drafted copies could be cast on two different
    turns in the same game - both occurrences count separately."""
    binder = _binder_with_alias("Owlbear", "1")
    metric = AverageTurnCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["user_turn_1_creatures_cast"] = "1"
    row["user_turn_3_creatures_cast"] = "1"

    metric.accumulate(row)
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for_arena_id(binder, "1")
    assert df.loc[owlbear_uuid, "sample_count"] == 2


def test_sample_count_is_occurrence_count_not_game_count(tmp_path: Path) -> None:
    binder = _binder_with_alias("Owlbear", "1")
    metric = AverageTurnCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )
    row = dict(_NA_ROW)
    row["user_turn_1_creatures_cast"] = "1"

    metric.accumulate(row)
    metric.accumulate(dict(_NA_ROW))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet").set_index("nocab_uuid")
    owlbear_uuid = _uuid_for_arena_id(binder, "1")
    assert df.loc[owlbear_uuid, "sample_count"] == 1


def test_card_never_cast_never_appears_in_output(tmp_path: Path) -> None:
    binder = _binder_with_alias("Owlbear", "1")
    metric = AverageTurnCastMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(dict(_NA_ROW))
    metric.finalize()

    df = pd.read_parquet(tmp_path / "out.parquet")
    assert df.shape[0] == 0
