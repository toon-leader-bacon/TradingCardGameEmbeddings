"""FactionInclusionTally: per-faction deck counts and per-(card,
faction) inclusion counts over guide decks, the shared state of
card_inclusion_metrics.py's two metrics.

The Gwent twin of ../fabtcg_decklists/hero_inclusion_tally.py, with the
deck's faction in place of its hero (a per-container copy: the two
differ in their condition and legality rule, and this project defers
cross-container dedup).
"""

from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.metrics.play_gwent.published_guide_decks import (
    GWENT_FACTIONS,
    GuideDeck,
    is_leader,
    legal_factions,
)
from src.schema.card import GenericCard


@dataclass(frozen=True)
class FactionInclusion:
    """One card's counts against one faction it is legal for.

    faction: a GWENT_FACTIONS member. legal_deck_count: decks of that
    faction. included_deck_count: those decks holding the card.
    """

    faction: str
    legal_deck_count: int
    included_deck_count: int


class FactionInclusionTally:
    """Counts decks per faction and, per card, decks per faction
    holding it."""

    def __init__(self) -> None:
        """Side effects: none."""
        self._decks_per_faction: Counter[str] = Counter()
        self._included: Counter[tuple[UUID, str]] = Counter()

    def add(self, guide_deck: GuideDeck) -> None:
        """Count one guide deck.

        Inputs: guide_deck (GuideDeck). Output: none.
        Side effects: updates this tally's counters.
        Exceptions: none.

        Example:
            >>> tally.add(guide_decks.guide_deck_for_row(row))
        """
        self._decks_per_faction[guide_deck.faction] += 1
        self._included.update(
            (card_uuid, guide_deck.faction) for card_uuid in guide_deck.card_uuids
        )

    def included_cards(self, card_lookup: CardLookup) -> list[GenericCard]:
        """Every card held by at least one counted deck, except leaders
        and the Unknown sentinel, sorted by uuid.

        The card universe of both metrics: a card no guide holds may
        postdate every guide (or be a token), so a 0 for it would not
        mean "never chosen".

        Inputs: card_lookup (the Gwent binder).
        Output: list[GenericCard].
        Side effects: none. Exceptions: none.

        Example:
            >>> len(tally.included_cards(binder))
            1100
        """
        uuids = sorted({card_uuid for card_uuid, _ in self._included}, key=str)
        cards = [card_lookup.get_by_uuid(card_uuid) for card_uuid in uuids]
        return [
            card
            for card in cards
            if card is not None
            and not is_leader(card)
            and card.name != CardBinder.UNKNOWN_CARD_NAME
        ]

    def legal_inclusions(self, card: GenericCard) -> list[FactionInclusion]:
        """card's counts against every faction it is legal for, in
        GWENT_FACTIONS order.

        Inputs: card. Output: list[FactionInclusion].
        Side effects: none. Exceptions: none.

        Example:
            >>> [i.faction for i in tally.legal_inclusions(tatterwing)]
            ['monster', 'syndicate']
        """
        legal = legal_factions(card)
        return [
            FactionInclusion(
                faction=faction,
                legal_deck_count=self._decks_per_faction[faction],
                included_deck_count=self._included[(card.nocab_uuid, faction)],
            )
            for faction in GWENT_FACTIONS
            if faction in legal
        ]
