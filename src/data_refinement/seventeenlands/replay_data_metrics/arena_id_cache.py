"""Caches replay_data's pipe-delimited Arena-ID hand cells into cards,
resolved for the life of one ReplayMetricScanner scan.

See src/data_refinement/README.md for this container's scope. Unlike
game_data_metrics/column_lookup.py (which resolves card identity
from the CSV HEADER, once, since game_data has one column per card),
replay_data has no per-card columns at all — the only per-card signal
lives in per-ROW CELL VALUES: opening_hand and candidate_hand_1 are
each a pipe-delimited string of Arena's numeric card IDs (e.g.
"104936|104917|105170"), resolved via
CardBinder.get_by_alias(source_game, DataSource.ARENA, id_str) instead
of by name.

ArenaIdCache is this container's equivalent of draft_data_metrics'
PickNameCache (see ../draft_data_metrics/pick_name_cache.py) — same
rationale, same shape: a small, fixed set of distinct cards (a few
hundred) gets referenced across potentially tens of millions of hand
cells in a multi-GB file, so ArenaIdCache caches every Arena ID's
lookup the first time it's seen and reuses it for the life of one
instance (one per ReplayMetricScanner.scan() call), instead of
re-querying CardBinder per cell.

Resolved data is returned as its own typed object, ResolvedHands
below, passed to a metric's accumulate() as an explicit second
argument (see replay_metric.py), never mutated
into the chunk — mirroring how game_data_metrics passes CardColumnSet
and draft_data_metrics passes its resolved_picks Series.

Only opening_hand and candidate_hand_1 are resolved today — the
remaining candidate_hand_2..7 columns exist in the raw file but have no
consumer yet (see replay_data_metrics/README.md for which metrics
exist). Add another ResolvedHands field and a corresponding lookup step
if a future metric needs one of them, rather than resolving every
candidate_hand_N column speculatively.
"""

from dataclasses import dataclass
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.lookup_cache import LookupCache
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


@dataclass(frozen=True)
class ResolvedHands:
    """One row's resolved hand data.

    The typed parameter every hand-based ReplayMetric's accumulate()
    receives alongside the raw chunk (see replay_metric.py) —
    mirrors game_data_metrics' CardColumnSet and draft_data_metrics'
    resolved_picks Series in spirit: resolved data passed explicitly,
    never mutated into the chunk under a magic column name.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    opening_hand: list[UUID]  # deduplicated resolved cards in the kept
    # opening hand
    candidate_hand_1: list[UUID]  # deduplicated resolved cards in the
    # first hand seen (pre-mulligan-decision)
    mulliganed: bool  # True if this game had at least one mulligan
    # (chunk["num_mulligans"] > 0) — derived, no CardBinder lookup
    # needed, but bundled here so a metric reads one resolved object
    # per row rather than mixing "resolved" and "raw chunk" lookups for
    # what is conceptually all "this row's hand-related facts"


class ArenaIdCache:
    """Caches CardBinder lookups for repeated Arena IDs across hand cells.

    Single-consumer to ReplayMetricScanner — one instance is
    constructed per scan() call and reused across every chunk of that
    scan, so its cache (and its accumulated unresolved_ids) persists
    for the life of the whole streamed pass. Mirrors
    ../draft_data_metrics/pick_name_cache.py's PickNameCache exactly in
    shape — see that module's docstring for the caching rationale,
    identical here.
    """

    def __init__(self, card_binder: CardBinder, source_game: GameId) -> None:
        """
        Inputs:
            card_binder: registry to resolve Arena IDs against.
            source_game: which game's alias namespace to resolve
                against (GameId.MTG for every 17lands source today).
        Output: none (constructor).
        Side effects: none — no CardBinder queries happen until
            get_hands() is called.
        Exceptions: none.
        """

        def lookup(arena_id: str) -> UUID | None:
            card = card_binder.get_by_alias(source_game, DataSource.ARENA, arena_id)
            return card.nocab_uuid if card is not None else None

        self._cache = LookupCache(lookup)

    def get_hands(self, chunk: pd.DataFrame) -> pd.Series:
        """Resolve one chunk's hand-related columns into a ResolvedHands Series.

        Reads chunk["opening_hand"], chunk["candidate_hand_1"], and
        chunk["num_mulligans"]; resolves each hand cell's Arena IDs via
        self._resolve_cell() (cache-first — see that method), and
        derives mulliganed directly from num_mulligans (no CardBinder
        involved for that field).

        Inputs:
            chunk: one chunk of the raw replay_data CSV — must include
                "opening_hand", "candidate_hand_1", and "num_mulligans"
                columns.
        Output: a pd.Series, index-aligned with chunk, one
            ResolvedHands per row.
        Side effects: mutates self._cache and self._unresolved_ids for
            any Arena ID not already cached (see _resolve_cell()).
        Exceptions: none expected from well-formed input.

        Example:
            >>> cache = ArenaIdCache(card_binder, GameId.MTG)
            >>> resolved = cache.get_hands(chunk)
        """
        opening_hands = chunk["opening_hand"].map(self._resolve_cell)
        candidate_hand_1s = chunk["candidate_hand_1"].map(self._resolve_cell)
        mulliganed = chunk["num_mulligans"] > 0

        return pd.Series(
            [
                ResolvedHands(
                    opening_hand=opening_hand,
                    candidate_hand_1=candidate_hand_1,
                    mulliganed=bool(is_mulliganed),
                )
                for opening_hand, candidate_hand_1, is_mulliganed in zip(
                    opening_hands, candidate_hand_1s, mulliganed
                )
            ],
            index=chunk.index,
        )

    def _resolve_cell(self, cell: str) -> list[UUID]:
        """Parse and resolve one pipe-delimited Arena-ID cell, cache-first.

        Private helper — single consumer is get_hands(), called once
        per hand cell. Splits cell on "|"; each distinct Arena ID is
        resolved at most once per ArenaIdCache instance — an ID
        already seen in an earlier call (this chunk or an earlier one)
        is served from the cache without querying card_binder again.
        An unresolvable ID is skipped from the returned list (recorded
        in the cache's unresolved_keys, not raised) — an occasional
        miss, e.g. a promo card absent from Scryfall's arena_id
        coverage, shouldn't abort an entire multi-GB scan. The returned
        list is deduplicated (order not guaranteed) — a card appearing
        twice in one hand (confirmed in real data, e.g. two copies of
        the same land) counts as ONE observed game for that card in a
        presence-triggered metric (see
        metrics/binary_trigger_rate/base.py), not two.

        Inputs:
            cell: one raw cell value — normally a pipe-delimited string
                of Arena numeric card IDs, or an empty string / NaN.
                Also handles a chunk where pandas has inferred this
                column as float64 instead of string (every non-null
                cell in this chunk's slice happens to be a single
                un-piped ID, e.g. a hand reduced to 1 card by repeated
                mulligans) — coerced via str(int(cell)) before parsing.
        Output: deduplicated list of resolved nocab_uuids. An
            empty/NaN cell yields [].
        Side effects: mutates the internal cache (always, per distinct
            ID not already cached).
        Exceptions: none expected from well-formed input.
        """
        if pd.isna(cell) or cell == "":
            return []
        if not isinstance(cell, str):
            # pandas infers each CSV chunk's column dtype independently —
            # a chunk where every non-null opening_hand/candidate_hand_1
            # cell happens to be a single un-piped ID (e.g. a hand
            # reduced to 1 card by repeated mulligans) gets read as
            # float64 instead of string, so cell arrives as e.g.
            # 104936.0 rather than "104936". Coerce before parsing —
            # same fix as cast_event_scanner.py's CastEventScanner.
            # _resolve_cell(), which hits this far more often (sparse
            # cast columns are single-ID far more frequently than hand
            # columns are single-card).
            cell = str(int(cell))

        resolved: list[UUID] = []
        seen: set[UUID] = set()
        for arena_id in cell.split("|"):
            nocab_uuid = self._cache.get(arena_id)
            if nocab_uuid is not None and nocab_uuid not in seen:
                resolved.append(nocab_uuid)
                seen.add(nocab_uuid)

        return resolved

    @property
    def unresolved_ids(self) -> frozenset[str]:
        """Every distinct Arena ID this cache has failed to resolve.

        Inputs: none (uses internal state).
        Output: every distinct Arena ID string get_hands() has been
            unable to resolve so far, across every call made to this
            instance. Bounded by the number of distinct cards ever
            referenced (a few hundred), not by row count. Returns a
            frozenset, not a sorted list like
            PickNameCache.unresolved_names — a deliberate deviation
            from that precedent, not an oversight: ReplayMetricScanner's
            checkpoint resume unions this with a restored set (see
            replay_metric_scanner.py's ReplayMetricScanResult docstring),
            which set semantics make trivial.
        Side effects: none.
        Exceptions: none.
        """
        return self._cache.unresolved_keys
