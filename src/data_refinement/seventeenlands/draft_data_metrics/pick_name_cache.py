"""Caches a draft_data CSV's 'pick' column values against CardBinder,
resolved to nocab_uuids.

See src/data_refinement/README.md for this container's scope. This is
draft_data_metrics's equivalent of game_data_metrics/column_lookup.py
— the one place that talks to CardBinder — but shaped differently on
purpose: game_data's card references live in the HEADER (fixed
`deck_<name>`-style columns, resolved once and reused unchanged across
every chunk). draft_data's card reference for a pick is a VALUE in the
data itself — the 'pick' column holds a card name string per row, and
the same handful of distinct names (a few hundred, for one set) repeat
across potentially millions of rows. Resolving via CardBinder on
every row (or even every chunk) would be wasted work, so
PickNameCache caches each name's resolution the first time it's
seen and reuses it for every later occurrence, for the life of one
cache instance (one per DraftMetricScanner.scan() call).

Uses name_lookup.py's find_uuid_by_name() for the actual resolution
policy (try CardBinder.get_by_name() first, fall back to
get_by_name_regex() for a multi-faced card, ambiguous match =
unresolved — the same policy column_lookup.py uses for header
resolution), composed with lookup_cache.py's LookupCache for the
caching mechanism itself (see tmp/REFACTOR.md §1 for why both were
extracted out of this file).
"""

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.lookup_cache import LookupCache
from src.data_refinement.seventeenlands.name_lookup import find_uuid_by_name
from src.schema.game_id import GameId


class PickNameCache:
    """Caches CardBinder lookups for repeated 'pick' column values.

    Single-consumer to DraftMetricScanner — one instance is constructed
    per scan() call and reused across every chunk of that scan, so its
    cache accumulates for the life of the whole streamed pass.
    """

    def __init__(self, card_binder: CardBinder, source_game: GameId) -> None:
        """
        Inputs:
            card_binder: registry to resolve pick names against.
            source_game: which game's cards to resolve against.
        Output: none (constructor).
        Side effects: none — no CardBinder queries happen until
            get_uuids() is called.
        Exceptions: none.
        """
        self._cache = LookupCache(
            lambda name: find_uuid_by_name(card_binder, source_game, name)
        )

    def get_uuids(self, pick_names: pd.Series) -> pd.Series:
        """Resolve one chunk's 'pick' column to a Series of nocab_uuids.

        Every distinct value in pick_names is resolved at most once
        per PickNameCache instance (see this module's docstring) —
        a name already seen in an earlier call, on an earlier chunk,
        is served from the cache without querying card_binder again.

        Inputs:
            pick_names: a chunk's 'pick' column — one card name string
                per row (e.g. chunk["pick"]).
        Output: a Series, index-aligned with pick_names, holding the
            resolved nocab_uuid for each row, or None where the name
            couldn't be resolved (see self.unresolved_names).
        Side effects: mutates the internal cache for any name not
            already cached.
        Exceptions: none expected from well-formed input.

        Example:
            >>> cache = PickNameCache(card_binder, GameId.MTG)
            >>> resolved = cache.get_uuids(chunk["pick"])
        """
        return pick_names.map(self._cache.get)

    @property
    def unresolved_names(self) -> list[str]:
        """Every distinct pick name this cache has failed to resolve.

        Inputs: none (uses internal state).
        Output: every distinct name get_uuids() has been unable to
            resolve so far, across every call made to this instance.
            Order not guaranteed.
        Side effects: none.
        Exceptions: none.
        """
        return sorted(self._cache.unresolved_keys)
