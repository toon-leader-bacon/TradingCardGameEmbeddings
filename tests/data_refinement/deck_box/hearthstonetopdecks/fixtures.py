"""Shared builders for the hearthstonetopdecks deck box tests."""

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _varint(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            result.append(byte | 0x80)
        else:
            result.append(byte)
            return bytes(result)


def encode_deckstring(
    card_counts: dict[int, int],
    hero_dbf_ids: tuple[int, ...] = (930,),
    format_id: int = 2,
    sideboard_counts: dict[tuple[int, int], int] | None = None,
) -> str:
    """A deck code in HearthSim's deckstring format (the inverse of
    deckstring.decode_deckstring), sideboard keys (owner, dbfId)."""
    values = [0, 1, format_id, len(hero_dbf_ids), *hero_dbf_ids]
    singles = [dbf_id for dbf_id, n in card_counts.items() if n == 1]
    doubles = [dbf_id for dbf_id, n in card_counts.items() if n == 2]
    multis = [(dbf_id, n) for dbf_id, n in card_counts.items() if n > 2]
    values += [len(singles), *singles, len(doubles), *doubles, len(multis)]
    for dbf_id, n in multis:
        values += [dbf_id, n]
    if sideboard_counts:
        values.append(1)
        side = sideboard_counts.items()
        side_singles = [(owner, d) for (owner, d), n in side if n == 1]
        side_doubles = [(owner, d) for (owner, d), n in side if n == 2]
        side_multis = [(owner, d, n) for (owner, d), n in side if n > 2]
        for block in (side_singles, side_doubles):
            values.append(len(block))
            for owner, dbf_id in block:
                values += [dbf_id, owner]
        values.append(len(side_multis))
        for owner, dbf_id, n in side_multis:
            values += [dbf_id, n, owner]
    return base64.b64encode(b"".join(_varint(v) for v in values)).decode()


def make_binder(dbf_ids_by_name: dict[str, list[int]]) -> CardBinder:
    """A Hearthstone binder holding one card per name, each registered
    under every listed dbfId."""
    binder = CardBinder()
    for name, dbf_ids in dbf_ids_by_name.items():
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.HEARTHSTONE,
            name=name,
            raw_content={"cost": 1},
            provenance=Provenance(
                data_source=DataSource.HEARTHSTONEJSON,
                source_id=str(dbf_ids[0]),
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        binder.create(card)
        for dbf_id in dbf_ids:
            binder.register_alias(
                GameId.HEARTHSTONE,
                DataSource.HEARTHSTONEJSON,
                str(dbf_id),
                card.nocab_uuid,
            )
    binder.ensure_unknown_card(GameId.HEARTHSTONE)
    return binder


def card_uuid(binder: CardBinder, dbf_id: int) -> UUID:
    card = binder.get_by_alias(
        GameId.HEARTHSTONE, DataSource.HEARTHSTONEJSON, str(dbf_id)
    )
    assert card is not None
    return card.nocab_uuid


def deck_row(slug: str, deck_code: str | None, title: str | None = "A Deck") -> dict:
    return {
        "url": f"https://www.hearthstonetopdecks.com/decks/{slug}/",
        "title": title,
        "deck_code": deck_code,
        "details": {"Format": "Standard"},
    }


def write_decks(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "decks.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path
