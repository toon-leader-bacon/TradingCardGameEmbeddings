"""HeroInclusionTally: per-hero deck counts and per-(card, hero)
inclusion counts over published FaB decklists, the shared state of
card_inclusion_metrics.py's two metrics.

Heroes are keyed by card name (HERO_NAMES is a list of names), with one
representative hero card per name kept for the legality check.
"""

from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.metrics.fabtcg_decklists.hero_legality import (
    is_hero,
    is_legal_for_hero,
)
from src.data_refinement.metrics.fabtcg_decklists.published_decklists import (
    FabDecklist,
)
from src.schema.card import GenericCard


@dataclass(frozen=True)
class HeroInclusion:
    """One card's counts against one hero it is legal for.

    hero_name: the hero's card name. legal_deck_count: decks playing
    that hero. included_deck_count: those decks holding the card.
    """

    hero_name: str
    legal_deck_count: int
    included_deck_count: int


class HeroInclusionTally:
    """Counts decks per hero and, per card, decks per hero holding it."""

    def __init__(self) -> None:
        """Side effects: none."""
        self._decks_per_hero: Counter[str] = Counter()
        self._included: Counter[tuple[UUID, str]] = Counter()
        self._hero_cards: dict[str, GenericCard] = {}

    def add(self, decklist: FabDecklist) -> None:
        """Count one decklist.

        Inputs: decklist (FabDecklist). Output: none.
        Side effects: updates this tally's counters.
        Exceptions: none.

        Example:
            >>> tally.add(decklists.decklist_for_slug(slug))
        """
        hero_name = decklist.hero.name
        self._hero_cards.setdefault(hero_name, decklist.hero)
        self._decks_per_hero[hero_name] += 1
        self._included.update(
            (card_uuid, hero_name) for card_uuid in decklist.card_uuids
        )

    def included_cards(self, card_lookup: CardLookup) -> list[GenericCard]:
        """Every card held by at least one counted deck, except heroes and
        the Unknown sentinel, sorted by uuid (a stable output order).

        The card universe of both metrics: a card no deck holds may not
        have been printed yet when these decks were played, so a 0 for
        it would not mean "never chosen".

        Inputs: card_lookup (the FaB binder).
        Output: list[GenericCard].
        Side effects: none. Exceptions: none.

        Example:
            >>> [card.name for card in tally.included_cards(binder)][:1]
            ['Command and Conquer']
        """
        uuids = sorted({card_uuid for card_uuid, _ in self._included}, key=str)
        cards = [card_lookup.get_by_uuid(card_uuid) for card_uuid in uuids]
        return [
            card
            for card in cards
            if card is not None
            and not is_hero(card)
            and card.name != CardBinder.UNKNOWN_CARD_NAME
        ]

    def legal_inclusions(self, card: GenericCard) -> list[HeroInclusion]:
        """card's counts against every counted hero it is legal for.

        Inputs: card (GenericCard).
        Output: one HeroInclusion per legal hero, in hero-name order.
        Side effects: none. Exceptions: none.

        Example:
            >>> tally.legal_inclusions(sink_below)[0].hero_name
            'Arakni, Marionette'
        """
        return [
            HeroInclusion(
                hero_name=hero_name,
                legal_deck_count=self._decks_per_hero[hero_name],
                included_deck_count=self._included[(card.nocab_uuid, hero_name)],
            )
            for hero_name, hero in sorted(self._hero_cards.items())
            if is_legal_for_hero(card, hero)
        ]
