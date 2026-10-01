"""CardGroup - how one isotropic metric column becomes a list of cards.

The isotropic multi-group and pick metrics name their card groups by
column: a DeckBox group uuid (kingdom_uuid, partial_deck_uuid,
deck_uuid_lo, ...) or a single card uuid (card_uuid). A CardGroup is a
Strategy that reads one such column of a row and returns the cards the
split may see, so GroupLabelDataConstructor and GroupPickDataConstructor
can assemble any [group_0, group_1] input from plain configuration.

Holdout: every card goes through the split's lookup, so a hidden card is
simply absent from its group.
"""

from dataclasses import dataclass, field
from typing import List, Protocol
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructors.row_values import (
    card_for_uuid,
    parsed_uuid,
)
from src.schema.card import GenericCard


class CardGroup(Protocol):
    """One row column -> the visible cards it names."""

    def cards(self, row: pd.Series, lookup: CardLookup) -> List[GenericCard]:
        """The row's card group, as the split sees it.

        Inputs: row (one metric row), lookup (the split's
            holdout-filtered lookup).
        Output: the visible cards; empty if nothing could be read.
        Side effects: may read a DeckBox. Exceptions: none.
        """
        ...


@dataclass(frozen=True)
class DeckColumnGroup:
    """The DeckBox group named by row[column].

    deck_box: the box the column's uuids point into; only read.
    column: the row column holding a group uuid.
    distinct: keep only the first copy of each card (a partial deck's
        distinct cards as options, say); False keeps copies.
    always_included: card uuids appended after the group's own cards
        when not already present (e.g. Dominion's base supply piles,
        which every kingdom offers but no kingdom group lists).
    """

    deck_box: DeckBox
    column: str
    distinct: bool = False
    always_included: tuple[UUID, ...] = field(default_factory=tuple)

    def cards(self, row: pd.Series, lookup: CardLookup) -> List[GenericCard]:
        """See CardGroup.cards. Order: the group's own order, then
        always_included; with distinct (or for always_included) a uuid
        already listed is skipped.

        Example:
            >>> DeckColumnGroup(box, "kingdom_uuid").cards(row, lookup)
            [<GenericCard Witch>, ...]
        """
        result: List[GenericCard] = []
        deck_uuid = parsed_uuid(row[self.column])
        deck = None if deck_uuid is None else self.deck_box.get_by_uuid(deck_uuid)
        if deck is None:
            return result

        # The group's own cards, then the always-included ones
        seen: set[UUID] = set()
        for card_uuid in deck.card_nocab_uuids:
            if self.distinct and card_uuid in seen:
                continue
            seen.add(card_uuid)
            _append_if_visible(result, lookup, card_uuid)
        for card_uuid in self.always_included:
            if card_uuid not in seen:
                seen.add(card_uuid)
                _append_if_visible(result, lookup, card_uuid)
        return result


@dataclass(frozen=True)
class CardColumnGroup:
    """A one-card group: the card named by row[column]."""

    column: str

    def cards(self, row: pd.Series, lookup: CardLookup) -> List[GenericCard]:
        """See CardGroup.cards: [the card], or [] if hidden/unparseable.

        Example:
            >>> CardColumnGroup("card_uuid").cards(row, lookup)
            [<GenericCard Witch>]
        """
        card = card_for_uuid(lookup, row[self.column])
        return [] if card is None else [card]


def _append_if_visible(
    cards: List[GenericCard], lookup: CardLookup, card_uuid: UUID
) -> None:
    """Append lookup's card for card_uuid to cards, if visible.

    Inputs: cards (the list being built), lookup, card_uuid.
    Output: none. Side effects: appends to cards (the caller's own,
        freshly built list). Exceptions: none.
    """
    card = lookup.get_by_uuid(card_uuid)
    if card is not None:
        cards.append(card)
