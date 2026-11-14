"""Typed decks from the MTG deck-site crawls (Archidekt, Moxfield,
Deckstats, MTGTop8): what each site's parser emits and the shared loop reads.

An entry carries every card id its site gives, so the matcher can try
the strongest first (mtg_card_matcher.py).
"""

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class MtgCardRef:
    """The fields that name one card: the matcher's cache key.

    oracle_id: Scryfall oracle id (Archidekt gives it).
    printing_id: Scryfall printing id (Moxfield, Deckstats give it).
    """

    name: str
    oracle_id: str | None
    printing_id: str | None


@dataclass(frozen=True)
class MtgDeckEntry:
    """count copies of one card (a CountedCard): the site's quantity or
    amount, at least 1 (parsers drop rows with fewer)."""

    count: int
    card: MtgCardRef

    def __post_init__(self) -> None:
        """Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError if count < 1."""
        if self.count < 1:
            raise ValueError(f"an entry holds at least 1 copy, not {self.count}")


@dataclass(frozen=True)
class CrawledDeck:
    """One site deck, as far as the deck box needs it: only the cards
    that count as the deck (see each site's deck_parser.py).

    source_id: the site's deck id, as a string.
    name: the deck's title; None when the site has none (the stage then
        names the deck by source_id).
    """

    source_id: str
    name: str | None
    entries: tuple[MtgDeckEntry, ...]

    @property
    def card_count(self) -> int:
        """Total copies across entries.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return sum(entry.count for entry in self.entries)


class DeckSkipReason(StrEnum):
    """Why a parser declined a whole deck. One list for every site: a
    new reason is added here."""

    UNLISTED = "unlisted"  # Archidekt unlisted, without include_unlisted
    NOT_A_DECK = "not_a_deck"  # Moxfield "package" card bundles


@dataclass(frozen=True)
class DeckSkip:
    """A parser's answer for a deck it declines."""

    reason: DeckSkipReason
