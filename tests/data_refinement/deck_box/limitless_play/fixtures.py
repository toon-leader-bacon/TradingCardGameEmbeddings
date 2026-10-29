"""Shared builders for the limitless_play deck box tests."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

SETS = [
    {"id": "sv4", "ptcgoCode": "PAR", "releaseDate": "2023/11/03"},
    {"id": "sv1", "ptcgoCode": "SVI", "releaseDate": "2023/03/31"},
    {"id": "sm115", "ptcgoCode": "HIF", "releaseDate": "2019/08/23"},
    {"id": "sma", "ptcgoCode": "HIF", "releaseDate": "2019/08/23"},
    {"id": "svp", "ptcgoCode": "PR-SV", "releaseDate": "2023/01/01"},
    {"id": "base1", "releaseDate": "1999/01/09"},
]


def make_binder(cards: list[tuple[str, str]]) -> CardBinder:
    """A Pokemon binder holding one card per (printing id, name)."""
    binder = CardBinder()
    for card_id, name in cards:
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.POKEMON,
            name=name,
            raw_content={"set": card_id.rsplit("-", 1)[0]},
            provenance=Provenance(
                data_source=DataSource.POKEMON_TCG,
                source_id=card_id,
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        binder.create(card)
        binder.register_alias(
            GameId.POKEMON, DataSource.POKEMON_TCG, card_id, card.nocab_uuid
        )
    binder.ensure_unknown_card(GameId.POKEMON)
    return binder


def card_uuid(binder: CardBinder, card_id: str) -> UUID:
    card = binder.get_by_alias(GameId.POKEMON, DataSource.POKEMON_TCG, card_id)
    assert card is not None
    return card.nocab_uuid


def raw_entry(count: int, set_code: str, number: str, name: str) -> dict:
    return {"count": count, "set": set_code, "number": number, "name": name}


def standings_row(entries: list[dict], archetype: str | None = "Charizard") -> dict:
    return {
        "player": "someone",
        "placing": 1,
        "deck": {"id": "x", "name": archetype, "icons": []} if archetype else None,
        "decklist": {"pokemon": entries, "trainer": [], "energy": []},
    }


def write_sets(tmp_path: Path) -> Path:
    path = tmp_path / "en.json"
    path.write_text(json.dumps(SETS), encoding="utf-8")
    return path


def write_standings(tmp_path: Path, tournaments: dict[str, list[dict]]) -> Path:
    path = tmp_path / "standings.jsonl"
    path.write_text(
        "".join(
            json.dumps({"tournament_id": tournament_id, "standings": rows}) + "\n"
            for tournament_id, rows in tournaments.items()
        ),
        encoding="utf-8",
    )
    return path
