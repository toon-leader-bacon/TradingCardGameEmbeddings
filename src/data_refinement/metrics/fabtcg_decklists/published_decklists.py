"""A read-only view of the published Flesh and Blood deck box
(data/final/decks/flesh_and_blood.db), keyed by fabtcg.com decklist slug.

Every metric in this package is driven by the raw decklist files
(data/raw/fabtcg_decklists/decklists/<slug>.html, see scanner.py), but
reads each deck's cards from the published box rather than re-resolving
card names: the box's extraction stage
(../../deck_box/fabtcg_decklists/extraction_stage.py) already resolved
them, and its deck uuid is the one every row here must point at. A
published deck's provenance source_id is its slug, so the slug is the
join key.

The box is only read (all_decks), never written: published deck boxes
are read-only for metrics.

The box keeps no slot structure, but a FaB hero is a card-intrinsic
fact (its typebox names Hero), so the hero is read off the deck's cards.
A deck with no hero, or more than one, has no FabDecklist (4,152 of the
4,161 published decks have exactly one; a Demi-Hero is not a hero).
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.fabtcg_decklists.hero_legality import is_hero
from src.schema.card import GenericCard, GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FabDecklist:
    """One published decklist with exactly one hero.

    deck_uuid: its uuid in the published box. hero: its hero card.
    card_uuids: every other distinct card in the deck (copies collapsed;
        the hero excluded).
    """

    deck_uuid: UUID
    hero: GenericCard
    card_uuids: frozenset[UUID]


class PublishedDecklists:
    """slug -> FabDecklist over one published FaB deck box, built once."""

    def __init__(self, deck_box: DeckBox, card_lookup: CardLookup) -> None:
        """
        Inputs:
            deck_box: the published FaB box; only read.
            card_lookup: the FaB binder the box was built from.
        Output: none (constructor).
        Side effects: reads every FaB deck in deck_box once; logs one
            INFO line with the deck counts, and a WARNING if two decks
            share a slug (the later one wins).
        Exceptions: none.

        Example:
            >>> decklists = PublishedDecklists(box, binder)
        """
        self._by_slug: dict[str, FabDecklist] = {}
        skipped = 0
        for deck in deck_box.all_decks(GameId.FLESH_AND_BLOOD):
            decklist = _decklist_for_deck(deck, card_lookup)
            if decklist is None or deck.provenance is None:
                skipped += 1
                continue
            if deck.provenance.source_id in self._by_slug:
                _logger.warning(
                    "PublishedDecklists: two decks share slug %r; keeping %s",
                    deck.provenance.source_id,
                    deck.nocab_uuid,
                )
            self._by_slug[deck.provenance.source_id] = decklist
        _logger.info(
            "PublishedDecklists: %d decklists with one hero, %d decks skipped",
            len(self._by_slug),
            skipped,
        )

    def decklist_for_slug(self, slug: str) -> FabDecklist | None:
        """The published decklist for one raw decklist file's slug.

        Inputs: slug (the raw file's stem).
        Output: its FabDecklist, or None if the box has no such deck or
            it has no single hero.
        Side effects: none. Exceptions: none.

        Example:
            >>> decklists.decklist_for_slug("gabe-sher-lexi-deck-calling-antwerp")
        """
        return self._by_slug.get(slug)

    def all_decklists(self) -> Iterable[FabDecklist]:
        """Every decklist in this view, in no particular order.

        Inputs: none. Output: Iterable[FabDecklist].
        Side effects: none. Exceptions: none.

        Example:
            >>> sum(1 for _ in decklists.all_decklists())
            4152
        """
        return self._by_slug.values()


def count_decks_per_hero(decklists: Iterable[FabDecklist]) -> dict[str, int]:
    """How many decklists play each hero, by hero card name - the
    tooling behind hero_labels.HERO_NAMES.

    Inputs: decklists (Iterable[FabDecklist]).
    Output: {hero card name: deck count}.
    Side effects: none. Exceptions: none.

    Example:
        >>> count_decks_per_hero(decklists.all_decklists())["Dorinthea Ironsong"]
        188
    """
    return dict(Counter(decklist.hero.name for decklist in decklists))


def _decklist_for_deck(
    deck: GenericDeck, card_lookup: CardLookup
) -> FabDecklist | None:
    """deck as a FabDecklist, if it has exactly one hero.

    Inputs: deck (a published FaB deck), card_lookup.
    Output: FabDecklist, or None for zero or several heroes. A card the
        lookup cannot find is dropped (it cannot be a hero either).
    Side effects: none. Exceptions: none.
    """
    cards = [
        card
        for card in map(card_lookup.get_by_uuid, set(deck.card_nocab_uuids))
        if card is not None
    ]
    heroes = [card for card in cards if is_hero(card)]
    if len(heroes) != 1:
        return None
    hero = heroes[0]
    return FabDecklist(
        deck_uuid=deck.nocab_uuid,
        hero=hero,
        card_uuids=frozenset(
            card.nocab_uuid for card in cards if card.nocab_uuid != hero.nocab_uuid
        ),
    )
