"""Shared builders for the MTG crawl deck box tests."""

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid5

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_NAMESPACE = UUID("2b8f9c1e-6d0a-4f7e-9a51-1c3e5d7f9b20")


def make_mtg_binder(cards: list[tuple[str, str, tuple[str, ...]]]) -> CardBinder:
    """An MTG binder holding one card per (name, oracle id, printing ids),
    each id registered as its alias, plus the Unknown sentinel. Uuids
    derive from the oracle id, so tests are reproducible."""
    binder = CardBinder()
    for name, oracle_id, printing_ids in cards:
        card = GenericCard(
            nocab_uuid=uuid5(_NAMESPACE, oracle_id),
            source_game=GameId.MTG,
            name=name,
            raw_content={"name": name},
            provenance=Provenance(
                data_source=DataSource.SCRYFALL,
                source_id=oracle_id,
                fetched_at=datetime(2026, 11, 8, tzinfo=timezone.utc),
            ),
        )
        binder.create(card)
        binder.register_alias(
            GameId.MTG, DataSource.SCRYFALL, oracle_id, card.nocab_uuid
        )
        for printing_id in printing_ids:
            binder.register_alias(
                GameId.MTG, DataSource.SCRYFALL_PRINTING, printing_id, card.nocab_uuid
            )
    binder.ensure_unknown_card(GameId.MTG)
    return binder


def card_uuid(oracle_id: str) -> UUID:
    """The uuid make_mtg_binder gives the card with oracle_id."""
    return uuid5(_NAMESPACE, oracle_id)


def write_shards(directory: Path, rows: list[dict]) -> Path:
    """rows as one gzipped crawl shard in directory."""
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / "decks-00000.jsonl.gz", "wt", encoding="utf-8") as file:
        file.write("".join(json.dumps(row) + "\n" for row in rows))
    return directory
