"""Shared in-memory Dominion cards and card groups for the isotropic dojo
tests - a CardBinder and DeckBox holding a tiny kingdom, so a
DataConstructor can be exercised without the real data files."""

from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def dominion_card(name: str) -> GenericCard:
    """A minimal Dominion GenericCard named name, with a fresh uuid."""
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.DOMINION,
        name=name,
        raw_content={"name": name},
        provenance=Provenance(
            data_source=DataSource.DOMINIONTABS,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def binder_and_group(names: list[str]) -> tuple[CardBinder, DeckBox, GenericDeck]:
    """A CardBinder holding one card per name, and a DeckBox holding one
    card group of all of them. Output: (binder, box, the stored group)."""
    binder = CardBinder()
    cards = [binder.create(dominion_card(name)) for name in names]
    box = DeckBox()
    group = box.create(
        GenericDeck(
            nocab_uuid=uuid4(),
            source_game=GameId.DOMINION,
            name="kingdom",
            card_nocab_uuids=[card.nocab_uuid for card in cards],
        )
    )
    return binder, box, group
