"""Stores a deck keyed by its card multiset, once.

Shared by the stages whose deck identity is the copy-count multiset
(deck_uuid_from_cards() in src/data_refinement/deck_ids.py): many raw
rows post the same list, so the first row stored names the deck and
every later identical list is a no-op.
"""

from datetime import datetime, timezone
from uuid import UUID

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_ids import deck_uuid_from_cards
from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def store_multiset_deck(
    box: DeckBox,
    source_game: GameId,
    name: str,
    card_nocab_uuids: list[UUID],
    data_source: DataSource,
    source_id: str,
) -> UUID | None:
    """create_if_absent() the deck under its multiset hash.

    create_if_absent() returns the deck passed in only when it was new
    (the stored one otherwise), so an identity check tells the two apart
    without a second lookup.

    Inputs: box, source_game, name, card_nocab_uuids (one per copy),
        data_source and source_id (the deck's provenance).
    Output: the deck uuid if newly created, else None.
    Side effects: may create one deck on box.
    Exceptions: as DeckBox.create_if_absent.

    Example:
        >>> store_multiset_deck(box, GameId.HEARTHSTONE, "Burn Mage", uuids,
        ...                     DataSource.HEARTHSTONETOPDECKS, "burn-mage")
        UUID('...')
    """
    deck = GenericDeck(
        nocab_uuid=deck_uuid_from_cards(card_nocab_uuids),
        source_game=source_game,
        name=name,
        card_nocab_uuids=card_nocab_uuids,
        provenance=Provenance(
            data_source=data_source,
            source_id=source_id,
            fetched_at=datetime.now(timezone.utc),
        ),
    )
    if box.create_if_absent(deck) is deck:
        return deck.nocab_uuid
    return None
