"""Generic cache-first "look up one key, remember misses" primitive shared
by every 17lands per-value CardBinder-lookup wrapper.

Extracted from three near-identical private implementations
(draft_data_metrics/pick_name_cache.py's PickNameCache._resolve_one_name,
replay_data_metrics/arena_id_cache.py's ArenaIdCache._resolve_cell,
replay_data_metrics/cast_event_scanner.py's CastEventScanner._resolve_cell)
that all did the same thing — check a dict, call CardBinder on a miss,
store the result (including None) unconditionally, track misses in a set —
differing only in which CardBinder method actually gets called
(get_by_name vs get_by_alias). See tmp/REFACTOR.md §1. Lives at this
seventeenlands/ shared level, not inside any one pipeline, since it's
needed by draft_data_metrics AND replay_data_metrics (PRINCIPLES.md
section 3: shared logic needed by more than one container belongs
somewhere visible to all of them, not buried in one).

LookupCache does not know what a "key" means (a card name, an Arena ID) or
how to look one up — the caller supplies that as a plain callback at
construction. This keeps LookupCache itself free of any 17lands-specific
policy: the exact-then-regex-fallback name policy lives in name_lookup.py,
composed with a LookupCache by PickNameCache, not folded into LookupCache
itself.
"""

from typing import Callable
from uuid import UUID


class LookupCache:
    """Memoizes key -> nocab_uuid lookups, cache-first, tracking misses.

    Single-consumer-per-instance — one LookupCache is constructed per
    resolver/cache object that needs one (PickNameCache, ArenaIdCache,
    each CastEventScanner instance), never shared across them — each
    resolver's cache state is independent by design (see
    cast_event_scanner.py's module docstring for why CastEventScanner
    doesn't share ArenaIdCache's cache instance).
    """

    def __init__(self, lookup: Callable[[str], UUID | None]) -> None:
        """
        Inputs:
            lookup: called at most once per distinct key, on a cache
                miss, to actually resolve that key (e.g. a closure
                calling CardBinder.get_by_alias(), or
                name_lookup.find_uuid_by_name()). May return None for
                an unresolvable key.
        Output: none (constructor).
        Side effects: none — lookup is never called until get() is.
        Exceptions: none.
        """
        self._lookup = lookup
        self._cache: dict[str, UUID | None] = {}
        self._unresolved_keys: set[str] = set()

    def get(self, key: str) -> UUID | None:
        """Resolve one key, cache-first.

        Checks the internal cache first; on a miss, calls the lookup
        callback given at construction and stores the result
        (including None) before returning, so this key is never
        looked up again for the life of this instance.

        Inputs:
            key: the value to resolve (a card name, an Arena ID
                string, etc. — opaque to LookupCache itself).
        Output: the resolved nocab_uuid, or None if unresolved.
        Side effects: mutates the internal cache (always) and the
            unresolved-keys set (only if this key is unresolved).
        Exceptions: none expected — propagates whatever the lookup
            callback itself raises, uncaught.

        Example:
            >>> cache = LookupCache(lambda name: find_uuid_by_name(card_binder, game, name))
            >>> cache.get("Lightning Bolt")
        """
        if key in self._cache:
            return self._cache[key]

        resolved = self._lookup(key)
        self._cache[key] = resolved
        if resolved is None:
            self._unresolved_keys.add(key)
        return resolved

    @property
    def unresolved_keys(self) -> frozenset[str]:
        """Every distinct key this instance has failed to resolve so far.

        Inputs: none (uses internal state).
        Output: every key get() has been unable to resolve, across
            every call made to this instance. Cumulative for the life
            of one instance.
        Side effects: none.
        Exceptions: none.
        """
        return frozenset(self._unresolved_keys)
