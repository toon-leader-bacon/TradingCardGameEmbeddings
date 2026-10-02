"""Shared builders for the fabtcg_decklists metric tests: FaB cards with a
typebox, a binder holding them, and a published-style deck box whose
decks carry their slug as provenance source_id."""

from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def fab_card(name: str, typebox: str, textbox: str = "") -> GenericCard:
    raw_content = {"name": name, "typebox": typebox}
    if textbox:
        raw_content["textbox"] = textbox
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.FLESH_AND_BLOOD,
        name=name,
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.CARDVAULT_FABTCG,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def fab_binder(cards: list[GenericCard]) -> CardBinder:
    binder = CardBinder()
    for card in cards:
        binder.create(card)
    return binder


def published_box(decks: dict[str, list[GenericCard]]) -> DeckBox:
    """An in-memory box with one deck per slug, its cards as given."""
    box = DeckBox()
    for slug, cards in decks.items():
        box.create(
            GenericDeck(
                nocab_uuid=uuid4(),
                source_game=GameId.FLESH_AND_BLOOD,
                name=f"fabtcg decklist {slug}",
                card_nocab_uuids=[card.nocab_uuid for card in cards],
                provenance=Provenance(
                    data_source=DataSource.FABTCG_DECKLISTS,
                    source_id=slug,
                    fetched_at=datetime.now(timezone.utc),
                ),
            )
        )
    return box
