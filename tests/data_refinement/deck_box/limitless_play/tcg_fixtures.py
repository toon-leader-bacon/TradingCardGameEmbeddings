"""Shared builders for the non-Pokemon Limitless deck box tests."""

from datetime import datetime, timezone
from uuid import UUID, uuid5

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_NAMESPACE = UUID("6f1c2a54-0d5e-4c55-9f0b-3c9e0b2d7a11")


def make_tcg_binder(game: GameId, cards: list[tuple[str, str | None]]) -> CardBinder:
    """A binder holding one card per (name, printed number or None),
    with the Unknown sentinel. Uuids are derived from the name and
    number, so tests are reproducible."""
    binder = CardBinder()
    for name, number in cards:
        card = GenericCard(
            nocab_uuid=uuid5(_NAMESPACE, f"{game.value}:{name}:{number}"),
            source_game=game,
            name=name,
            raw_content={"name": name},
            provenance=Provenance(
                data_source=DataSource.TCGCSV,
                source_id=f"{name}:{number}",
                fetched_at=datetime(2026, 11, 8, tzinfo=timezone.utc),
            ),
        )
        binder.create(card)
        if number is not None:
            binder.register_alias(
                game, DataSource.PRINTED_NUMBER, number, card.nocab_uuid
            )
    binder.ensure_unknown_card(game)
    return binder


def card_uuid_named(binder: CardBinder, game: GameId, name: str) -> UUID:
    """The uuid of the single binder card called name."""
    (card,) = binder.get_by_name(game, name)
    return card.nocab_uuid


def leader_entry(name: str, set_code: str, number: str) -> dict:
    """A single-card section's raw entry (no count)."""
    return {"name": name, "set": set_code, "number": number}
