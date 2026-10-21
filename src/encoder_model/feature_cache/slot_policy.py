"""SlotPolicy: which offered keys a feature store admits, and which it
evicts to make room. Strategy, so admission and eviction swap without
touching the store.

Eviction policies: FillUntilFull (never evicts), FirstInFirstOut,
LeastRecentlyUsed and LeastFrequentlyUsed. SecondOfferAdmission is a
Decorator that adds an admission filter to any of them. Every policy
honours pins: the store passes its pinned keys, and a policy never
evicts one.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import AbstractSet, Protocol, Sequence

from src.encoder_model.feature_cache.card_text_key import CardTextKey


@dataclass(frozen=True)
class AdmissionDecision:
    """A policy's answer to one put.

    admitted: the offered keys to store, in offered order.
    evicted: held keys to drop; their rows are reused for admitted keys.
        A store checks len(admitted) <= free rows + len(evicted), and
        that no evicted key is pinned.
    refused_for_pins: offered keys this policy would have admitted had
        no held row been pinned; they pass through unstored.
    """

    admitted: list[CardTextKey]
    evicted: list[CardTextKey]
    refused_for_pins: int = 0


class SlotPolicy(Protocol):
    """Strategy: admission and eviction for one store.

    A store refuses a malformed decision with RuntimeError after admit()
    has run, so a stateful policy's records may no longer match the store:
    treat that error as fatal, never retry past it."""

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        """Inputs: keys a lookup just found. Output: none.
        Side effects: updates the policy's own recency/frequency state.
        Exceptions: none."""
        ...

    def admit(
        self,
        offered: Sequence[CardTextKey],
        free_rows: int,
        pinned: AbstractSet[CardTextKey],
    ) -> AdmissionDecision:
        """Inputs: offered (keys not held, distinct), free_rows (>= 0),
        pinned (keys with a pin; may include keys not held).
        Output: AdmissionDecision admitting at most free_rows plus the
        held keys not in pinned, and evicting only keys not in pinned.
        Side effects: updates the policy's state for what it admits and
        evicts. Exceptions: none."""
        ...


class FillUntilFull:
    """SlotPolicy: admit while there are free rows, never evict. Exact and
    the cheapest when every key fits; once full, new keys pass through
    unstored."""

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        """Nothing to record: this policy never evicts. Inputs: keys.
        Output: none. Side effects: none. Exceptions: none."""

    def admit(
        self,
        offered: Sequence[CardTextKey],
        free_rows: int,
        pinned: AbstractSet[CardTextKey],
    ) -> AdmissionDecision:
        """The first free_rows offered keys; nothing evicted. Pins never
        limit it, since it never evicts.

        Inputs: offered, free_rows (>= 0), pinned (unused).
        Output: AdmissionDecision. Side effects: none. Exceptions: none.

        Example:
            >>> FillUntilFull().admit([key_a, key_b], free_rows=1, pinned=set()).admitted
            [key_a]
        """
        return AdmissionDecision(admitted=list(offered[:free_rows]), evicted=[])


class _EvictingPolicy(ABC):
    """Template Method base for the policies that evict: admit every
    offered key, in order, up to the store's capacity, evicting unpinned
    held keys to make room. When more keys are offered than the store
    holds, the first capacity_rows win.

    Subclasses keep their own record of the held keys and say which to
    evict (_pop_unpinned_victims) and how a hit or an admission changes
    that record. Tracks the keys it admitted, so it assumes it is the only
    policy deciding for its store."""

    def __init__(self, capacity_rows: int) -> None:
        """Inputs: capacity_rows (>= 1, the store's). Output: none
        (constructor). Side effects: none.
        Exceptions: ValueError if capacity_rows < 1."""
        if capacity_rows < 1:
            raise ValueError(f"capacity_rows must be >= 1, got {capacity_rows}")
        self._capacity_rows = capacity_rows

    def admit(
        self,
        offered: Sequence[CardTextKey],
        free_rows: int,
        pinned: AbstractSet[CardTextKey],
    ) -> AdmissionDecision:
        """The first capacity_rows offered keys, as many as the free rows
        and the unpinned held rows make room for, evicting the subclass's
        victims beyond the free rows.

        Inputs: offered (not held, distinct), free_rows (>= 0), pinned.
        Output: AdmissionDecision; refused_for_pins counts the keys cut
            because pinned rows could not be evicted.
        Side effects: records the admitted keys and forgets the evicted
            ones. Exceptions: none.

        Example:
            >>> policy = LeastRecentlyUsed(2)
            >>> policy.admit([key_a, key_b, key_c], free_rows=2, pinned=set()).admitted
            [key_a, key_b]
        """
        wanted = list(offered[: self._capacity_rows])

        # Room is the free rows plus every held row no pin protects
        held = self._held_count()
        room = free_rows + held - self._count_held_among(pinned)
        result_admitted = wanted[:room]
        unpinned_room = min(len(wanted), free_rows + held)
        refused_for_pins = unpinned_room - len(result_admitted)

        # Evict only what the free rows cannot cover
        shortfall = max(0, len(result_admitted) - free_rows)
        evicted = self._pop_unpinned_victims(shortfall, pinned)
        self._record_admitted(result_admitted)
        return AdmissionDecision(result_admitted, evicted, refused_for_pins)

    @abstractmethod
    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        """As SlotPolicy.record_hits."""

    @abstractmethod
    def _held_count(self) -> int:
        """How many keys this policy holds. Output: int. Side effects:
        none. Exceptions: none."""

    @abstractmethod
    def _holds(self, key: CardTextKey) -> bool:
        """Whether this policy holds key. Side effects: none."""

    @abstractmethod
    def _pop_unpinned_victims(
        self, count: int, pinned: AbstractSet[CardTextKey]
    ) -> list[CardTextKey]:
        """Remove and return count held keys not in pinned, the ones this
        policy evicts first. Inputs: count (<= held keys not in pinned),
        pinned. Output: list[CardTextKey]. Side effects: forgets them.
        Exceptions: none."""

    @abstractmethod
    def _record_admitted(self, keys: Sequence[CardTextKey]) -> None:
        """Start holding keys (newly admitted, not held). Side effects:
        updates the record. Exceptions: none."""

    def _count_held_among(self, pinned: AbstractSet[CardTextKey]) -> int:
        """How many pinned keys this policy holds. Iterates pinned, which
        is a few batches' keys, not the held keys, which can be many.

        Private helper - single consumer is admit.
        Inputs: pinned. Output: int. Side effects: none. Exceptions: none.
        """
        return sum(1 for key in pinned if self._holds(key))


class FirstInFirstOut(_EvictingPolicy):
    """SlotPolicy: evict the oldest admitted unpinned keys. Its own
    record_hits ignores hits (LeastRecentlyUsed overrides that). A natural
    choice for the RAM tier, which never sees the hits the VRAM tier
    answers."""

    def __init__(self, capacity_rows: int) -> None:
        """Inputs: capacity_rows (>= 1, the store's). Output: none
        (constructor). Side effects: none.
        Exceptions: ValueError if capacity_rows < 1.

        Example:
            >>> FirstInFirstOut(1000)
        """
        super().__init__(capacity_rows)
        # Held keys, oldest (first to evict) first
        self._order: OrderedDict[CardTextKey, None] = OrderedDict()

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        """Nothing to record: admission order alone decides. Inputs: keys.
        Output: none. Side effects: none. Exceptions: none."""

    def _held_count(self) -> int:
        return len(self._order)

    def _holds(self, key: CardTextKey) -> bool:
        return key in self._order

    def _pop_unpinned_victims(
        self, count: int, pinned: AbstractSet[CardTextKey]
    ) -> list[CardTextKey]:
        """The count oldest keys not in pinned: a loop from the old end
        that skips pinned keys."""
        result: list[CardTextKey] = []
        if count == 0:
            return result
        for key in self._order:
            if key not in pinned:
                result.append(key)
                if len(result) == count:
                    break
        for key in result:
            del self._order[key]
        return result

    def _record_admitted(self, keys: Sequence[CardTextKey]) -> None:
        """Append keys at the newest end, in order."""
        for key in keys:
            self._order[key] = None


class LeastRecentlyUsed(FirstInFirstOut):
    """SlotPolicy: a FirstInFirstOut whose hits move a key to the newest
    end, so it evicts the least recently hit or admitted unpinned keys.
    The default."""

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        """Mark held keys as the most recently used, in order. Inputs:
        keys. Output: none. Side effects: reorders the record.
        Exceptions: none.

        Example:
            >>> policy.record_hits([key_a])
        """
        for key in keys:
            if key in self._order:
                self._order.move_to_end(key)


class LeastFrequentlyUsed(_EvictingPolicy):
    """SlotPolicy: evict the held keys with the fewest hits, oldest first
    among ties. A key starts at 0 hits when admitted, and its count is
    forgotten when it is evicted. Keeps staples over the long tail, as in
    contrastive diets.

    Frequency buckets (hit count -> keys in admission or last-hit order)
    keep record_hits and admit O(1) per key, plus one pass over the
    distinct counts per eviction."""

    def __init__(self, capacity_rows: int) -> None:
        """Inputs: capacity_rows (>= 1, the store's). Output: none
        (constructor). Side effects: none.
        Exceptions: ValueError if capacity_rows < 1.

        Example:
            >>> LeastFrequentlyUsed(1000)
        """
        super().__init__(capacity_rows)
        self._hits_of: dict[CardTextKey, int] = {}
        self._keys_by_hits: dict[int, OrderedDict[CardTextKey, None]] = {}

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        """Count one hit for each held key (a key repeated in keys counts
        each time). Inputs: keys. Output: none. Side effects: moves each
        key up one bucket. Exceptions: none.

        Example:
            >>> policy.record_hits([key_a, key_a])
        """
        for key in keys:
            if self._holds(key):
                self._move_up_a_bucket(key)

    def _held_count(self) -> int:
        return len(self._hits_of)

    def _holds(self, key: CardTextKey) -> bool:
        return key in self._hits_of

    def _pop_unpinned_victims(
        self, count: int, pinned: AbstractSet[CardTextKey]
    ) -> list[CardTextKey]:
        """The count least-hit keys not in pinned: the buckets in rising
        hit order, each oldest first, skipping pinned keys; empty buckets
        are dropped."""
        result: list[CardTextKey] = []
        # Choose the victims, least hit first
        for hits in sorted(self._keys_by_hits):
            if len(result) == count:
                break
            for key in self._keys_by_hits[hits]:
                if key not in pinned:
                    result.append(key)
                    if len(result) == count:
                        break

        # Forget them, and any bucket they leave empty
        for key in result:
            hits = self._hits_of.pop(key)
            bucket = self._keys_by_hits[hits]
            del bucket[key]
            if not bucket:
                del self._keys_by_hits[hits]
        return result

    def _record_admitted(self, keys: Sequence[CardTextKey]) -> None:
        """Put keys in the 0-hit bucket, in order."""
        if not keys:
            return
        bucket = self._keys_by_hits.setdefault(0, OrderedDict())
        for key in keys:
            self._hits_of[key] = 0
            bucket[key] = None

    def _move_up_a_bucket(self, key: CardTextKey) -> None:
        """Move a held key from its bucket to the next one up, at the
        newest end; drop the bucket it left if empty.

        Private helper - single consumer is record_hits.
        Inputs: key (held). Output: none. Side effects: as above.
        Exceptions: none.
        """
        hits = self._hits_of[key]
        bucket = self._keys_by_hits[hits]
        del bucket[key]
        if not bucket:
            del self._keys_by_hits[hits]
        self._hits_of[key] = hits + 1
        self._keys_by_hits.setdefault(hits + 1, OrderedDict())[key] = None


class SecondOfferAdmission:
    """SlotPolicy Decorator: pass a key on to the inner policy only on its
    second offer. A first offer is remembered and the key passes through
    unstored, so a text seen once (a one-off noise draw, a rare card)
    takes no row and evicts nothing.

    The memory of first offers is bounded: a FIFO set of remembered_keys,
    the oldest forgotten first. A second offer the inner policy refuses
    (past its capacity, or for pins) stays remembered in its old place,
    so its next offer is still a repeat unless newer first offers have
    pushed it out. Under prefetch, a card's first
    reservation encodes it without admitting, so its step encodes it
    again."""

    def __init__(self, inner: SlotPolicy, remembered_keys: int) -> None:
        """Inputs: inner (the policy that admits and evicts),
        remembered_keys (>= 1; how many first offers to remember, usually
        the tier's capacity_rows). Output: none (constructor).
        Side effects: none. Exceptions: ValueError if remembered_keys < 1.

        Example:
            >>> SecondOfferAdmission(LeastRecentlyUsed(1000), remembered_keys=1000)
        """
        if remembered_keys < 1:
            raise ValueError(f"remembered_keys must be >= 1, got {remembered_keys}")
        self._inner = inner
        self._remembered_keys = remembered_keys
        # Keys offered once and not yet again, oldest first
        self._offered_once: OrderedDict[CardTextKey, None] = OrderedDict()

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        """The inner policy's record_hits. Inputs: keys. Output: none.
        Side effects: the inner policy's. Exceptions: none."""
        self._inner.record_hits(keys)

    def admit(
        self,
        offered: Sequence[CardTextKey],
        free_rows: int,
        pinned: AbstractSet[CardTextKey],
    ) -> AdmissionDecision:
        """The inner policy's decision over the offered keys seen before;
        the rest are remembered and pass through.

        Inputs: offered (not held, distinct), free_rows, pinned.
        Output: the inner policy's AdmissionDecision (its admitted keys
            are all second offers; its refused_for_pins as it counts).
        Side effects: remembers first offers (forgetting the oldest past
            remembered_keys), forgets the second offers the inner policy
            admitted; the inner policy's.
        Exceptions: none.

        Example:
            >>> policy = SecondOfferAdmission(LeastRecentlyUsed(2), remembered_keys=2)
            >>> policy.admit([key_a], free_rows=2, pinned=set()).admitted
            []
            >>> policy.admit([key_a], free_rows=2, pinned=set()).admitted
            [key_a]
        """
        repeats: list[CardTextKey] = []
        first_offers: list[CardTextKey] = []

        # Sort the whole offer before remembering, so a first offer cannot
        # push out a repeat later in the same offer
        for key in offered:
            if key in self._offered_once:
                repeats.append(key)
            else:
                first_offers.append(key)
        for key in first_offers:
            self._remember(key)

        # Only the repeats reach the inner policy; forget the admitted ones
        if not repeats:
            return AdmissionDecision(admitted=[], evicted=[])
        result = self._inner.admit(repeats, free_rows, pinned)
        self._forget(result.admitted)
        return result

    def _forget(self, keys: Sequence[CardTextKey]) -> None:
        """Drop keys from the first-offer memory. A key may already be
        gone: a later first offer in the same admit can push it out.

        Private helper - single consumer is admit.
        Inputs: keys. Output: none. Side effects: as above.
        Exceptions: none.
        """
        for key in keys:
            self._offered_once.pop(key, None)

    def _remember(self, key: CardTextKey) -> None:
        """Remember a first offer at the newest end, forgetting the oldest
        once more than remembered_keys are held. Only first offers are
        added here: a repeat the inner policy refuses keeps its old place.

        Private helper - single consumer is admit.
        Inputs: key (not remembered). Output: none. Side effects: as above.
        Exceptions: none.
        """
        self._offered_once[key] = None
        if len(self._offered_once) > self._remembered_keys:
            self._offered_once.popitem(last=False)
