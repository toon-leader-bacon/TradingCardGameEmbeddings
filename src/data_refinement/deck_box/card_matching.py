"""Turning a source's card references into binder cards, with a cache
and match counts. Shared by every deck stage that matches cards in
several steps (limitless_play/, mtg_crawls/).

A source defines a frozen card reference (the fields that name one card,
no copy count) and a CachedCardMatcher subclass whose _match() tries its
steps. CachedCardMatcher adds the rest:

- a cache, so each distinct reference is looked up once per run;
- match_counts (which step matched, per decklist entry) and
  unmatched_counts (the references nothing matched), for the stage's
  summary log.

card_uuids_per_copy() then expands a decklist into one uuid per copy,
the Unknown sentinel standing in for a miss.

Example (a source with a name-only reference):

    @dataclass(frozen=True)
    class NameRef:
        name: str

    class NameMatcher(CachedCardMatcher[NameRef]):
        def _match(self, key):
            uuid = uuid_for_name_or_front_face(lookup, game, key.name)
            if uuid is None:
                return None, CardMatchKind.UNMATCHED
            return uuid, CardMatchKind.NAME
"""

from abc import ABC, abstractmethod
from collections import Counter
from enum import StrEnum
from typing import Generic, Hashable, Iterable, Protocol, TypeVar
from uuid import UUID

KeyT = TypeVar("KeyT", bound=Hashable)
KeyT_co = TypeVar("KeyT_co", bound=Hashable, covariant=True)


class CardMatchKind(StrEnum):
    """Which step matched a card (for a stage's summary log)."""

    CARD_ID = "card_id"  # a source id naming the card itself (an oracle id)
    PRINTING = "printing"  # a source id naming one printing of the card
    NAME = "name"
    UNMATCHED = "unmatched"


class CachedCardMatcher(ABC, Generic[KeyT]):
    """Template Method: a card reference -> its binder card's nocab_uuid,
    cached and counted. A subclass supplies only _match.
    """

    def __init__(self) -> None:
        """
        Inputs: none. Output: none (constructor).
        Side effects: none. Exceptions: none.
        """
        self._match_cache: dict[KeyT, tuple[UUID | None, CardMatchKind]] = {}
        self.match_counts: Counter[CardMatchKind] = Counter()
        self.unmatched_counts: Counter[KeyT] = Counter()

    def match_card_uuid(self, key: KeyT) -> UUID | None:
        """The binder card key names, or None if no step matches.

        Inputs: key. Output: nocab_uuid or None.
        Side effects: fills the cache; counts each call's match kind in
            match_counts, and each unmatched key in unmatched_counts
            (cache hits included, so the counts are per decklist entry).
        Exceptions: whatever _match raises.

        Example:
            >>> matcher.match_card_uuid(NameRef("Lightning Bolt"))
            UUID('...')
        """
        if key not in self._match_cache:
            self._match_cache[key] = self._match(key)
        result, match_kind = self._match_cache[key]
        self.match_counts[match_kind] += 1
        if match_kind is CardMatchKind.UNMATCHED:
            self.unmatched_counts[key] += 1
        return result

    @abstractmethod
    def _match(self, key: KeyT) -> tuple[UUID | None, CardMatchKind]:
        """One key's lookup, uncached.

        Inputs: key. Output: (nocab_uuid, the step that matched), or
            (None, UNMATCHED).
        Side effects: none. Exceptions: as the subclass documents.
        """


class CountedCard(Protocol[KeyT_co]):
    """A decklist entry: count copies of the card key names."""

    @property
    def count(self) -> int:
        """Copies of the card."""
        ...

    @property
    def card(self) -> KeyT_co:
        """The card's reference."""
        ...


def card_uuids_per_copy(
    entries: Iterable[CountedCard[KeyT]],
    matcher: CachedCardMatcher[KeyT],
    unknown_uuid: UUID,
) -> list[UUID]:
    """A decklist as one binder uuid per copy, the Unknown sentinel
    standing in for an entry nothing matched.

    Inputs: entries, matcher, unknown_uuid (the game's sentinel).
    Output: list of uuids, len == the entries' summed counts, in entry
        order.
    Side effects: as matcher.match_card_uuid.
    Exceptions: as matcher.match_card_uuid.

    Example:
        >>> card_uuids_per_copy([MtgDeckEntry(4, bolt)], matcher, unknown)
        [UUID('...'), UUID('...'), UUID('...'), UUID('...')]
    """
    result: list[UUID] = []
    for entry in entries:
        card_uuid = matcher.match_card_uuid(entry.card)
        result.extend([card_uuid or unknown_uuid] * entry.count)
    return result
