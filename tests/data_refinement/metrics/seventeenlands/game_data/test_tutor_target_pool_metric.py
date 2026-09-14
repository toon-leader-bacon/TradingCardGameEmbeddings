"""Skeleton-stage test stubs for tutor_target_pool_metric.py's
TutorTargetPoolMetric - the fan-out (one row per pool card) streaming
shape. Bodies are filled in by design-recipe-implement.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_pool_metric import (
    TutorTargetPoolMetric,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


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


def _binder_with_cards(names: list[str]) -> CardBinder:
    binder = CardBinder()
    for name in names:
        binder.create(_make_card(name))
    return binder


_HEADER = [
    "draft_id",
    "match_number",
    "game_number",
    "deck_Owlbear",
    "sideboard_Owlbear",
    "tutored_Owlbear",
    "deck_Goblin Morningstar",
    "sideboard_Goblin Morningstar",
    "tutored_Goblin Morningstar",
]


def _row(
    owlbear_deck: int = 0,
    owlbear_sideboard: int = 0,
    owlbear_tutored: int = 0,
    morningstar_deck: int = 0,
    morningstar_sideboard: int = 0,
    morningstar_tutored: int = 0,
) -> dict:
    return {
        "draft_id": "draft1",
        "match_number": 0,
        "game_number": 0,
        "deck_Owlbear": owlbear_deck,
        "sideboard_Owlbear": owlbear_sideboard,
        "tutored_Owlbear": owlbear_tutored,
        "deck_Goblin Morningstar": morningstar_deck,
        "sideboard_Goblin Morningstar": morningstar_sideboard,
        "tutored_Goblin Morningstar": morningstar_tutored,
    }


def _uuid_for(binder: CardBinder, name: str) -> str:
    cards = binder.get_by_name(GameId.MTG, name)
    assert len(cards) == 1
    return str(cards[0].nocab_uuid)


def test_writes_one_row_per_pool_card_deck_union_sideboard(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
    metric = TutorTargetPoolMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row(owlbear_deck=4, morningstar_sideboard=2))
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    rows = table.to_pylist()
    pool_card_uuids = {row["pool_card_uuid"] for row in rows}
    assert pool_card_uuids == {
        _uuid_for(binder, "Owlbear"),
        _uuid_for(binder, "Goblin Morningstar"),
    }
    for row in rows:
        assert row["draft_id"] == "draft1"
        assert row["match_number"] == 0
        assert row["game_number"] == 0


def test_tutored_flag_true_only_for_cards_present_in_tutored_columns(
    tmp_path: Path,
) -> None:
    binder = _binder_with_cards(["Owlbear", "Goblin Morningstar"])
    metric = TutorTargetPoolMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(
        _row(
            owlbear_deck=4, owlbear_tutored=1, morningstar_deck=4, morningstar_tutored=0
        )
    )
    metric.finalize()

    rows = pq.read_table(tmp_path / "out.parquet").to_pylist()
    by_uuid = {row["pool_card_uuid"]: row["tutored"] for row in rows}
    assert by_uuid[_uuid_for(binder, "Owlbear")] is True
    assert by_uuid[_uuid_for(binder, "Goblin Morningstar")] is False


def test_empty_pool_writes_zero_rows(tmp_path: Path) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = TutorTargetPoolMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    metric.accumulate(_row())  # every column zero - empty pool
    metric.finalize()

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.num_rows == 0


def test_card_in_both_deck_and_sideboard_appears_once_not_twice(
    tmp_path: Path,
) -> None:
    binder = _binder_with_cards(["Owlbear"])
    metric = TutorTargetPoolMetric(
        binder, _HEADER, GameId.MTG, output_path=tmp_path / "out.parquet"
    )

    # A card can't literally be in both deck and sideboard in one real
    # game, but the pool union logic should still dedupe defensively.
    metric.accumulate(_row(owlbear_deck=4, owlbear_sideboard=2))
    metric.finalize()

    rows = pq.read_table(tmp_path / "out.parquet").to_pylist()
    owlbear_uuid = _uuid_for(binder, "Owlbear")
    assert sum(1 for row in rows if row["pool_card_uuid"] == owlbear_uuid) == 1


def test_never_hashes_pool_as_a_deck_uuid() -> None:
    """This metric's identity is (draft_id, match_number, game_number,
    pool_card_uuid), never a deck_uuid_from_cards() hash over the wider
    deck-union-sideboard pool - see module docstring."""
    assert not hasattr(TutorTargetPoolMetric, "_deck_box")
    assert "deck_box" not in TutorTargetPoolMetric.__init__.__annotations__
