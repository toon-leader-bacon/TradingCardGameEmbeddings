"""Shared, non-test helpers for isotropic/summary's test files - builds
a minimal real dominiontabs CardBinder (so every test resolves card
names through the actual DominionTabsCardIngestionStage, not a fake
lookup) and small Flavor A row fixtures.
"""

import json
from pathlib import Path
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.dominiontabs.ingestion_stage import (
    DominionTabsCardIngestionStage,
)
from src.schema.game_id import GameId


def binder_from_card_names(
    card_names: list[str],
    tmp_path: Path,
    card_types: dict[str, list[str]] | None = None,
) -> CardBinder:
    """Ingest one minimal dominiontabs card per name into a fresh CardBinder.

    Every card is an ["Action"] unless card_types names its types."""
    card_types = card_types or {}
    binder = CardBinder()
    raw_dir = tmp_path / "dominiontabs_raw"
    raw_dir.mkdir(exist_ok=True)
    cards_db = [
        {"card_tag": name, "types": card_types.get(name, ["Action"]), "cost": "3"}
        for name in card_names
    ]
    cards_en_us = {name: {"name": name, "description": ""} for name in card_names}
    (raw_dir / "cards_db.json").write_text(json.dumps(cards_db), encoding="utf-8")
    (raw_dir / "cards_en_us.json").write_text(json.dumps(cards_en_us), encoding="utf-8")
    DominionTabsCardIngestionStage().ingest(raw_dir, binder)
    return binder


def card_uuid(binder: CardBinder, name: str) -> UUID:
    card = binder.get_by_name_single(GameId.DOMINION, name)
    assert card is not None
    return card.nocab_uuid


def player_entry(
    nick: str,
    rank: int,
    end_deck: dict[str, int] | None = None,
    turns: int = 20,
    resigned: bool = False,
) -> dict:
    entry: dict = {"nick": nick, "rank": rank, "turns": turns}
    if end_deck is not None:
        entry["end"] = {"deck": end_deck, "vps": {}}
    if resigned:
        entry["resigned"] = True
    return entry


def summary_row(supply: list[str], players: list[dict], **overrides: object) -> dict:
    row: dict = {
        "board": {"supply": supply},
        "players": players,
        "serial": 1,
        "start_time": 0,
        "end_time": 1,
    }
    row.update(overrides)
    return row
