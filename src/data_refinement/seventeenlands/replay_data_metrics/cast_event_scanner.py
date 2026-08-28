"""Scans replay_data's turn-indexed cast columns for per-instance cast events.

See arena_id_cache.py's module docstring for the general per-cell
Arena-ID resolution background this builds on — ArenaIdCache resolves
the 2 fixed hand columns; CastEventScanner here resolves the much
wider, sparser set of turn-indexed "was card X cast this turn"
columns, confirmed (this session, against real data) to number up to 4
column families per side per turn:
  - <side>_turn_T_creatures_cast
  - <side>_turn_T_non_creatures_cast
  - <side>_turn_T_<side>_instants_sorceries_cast        (own-turn)
  - <other_side>_turn_T_<side>_instants_sorceries_cast  (off-turn/
    flash-speed — confirmed real via oppo_turn_3_user_instants_sorceries_cast
    genuinely recording the user casting during the opponent's own
    turn 3)
across up to 30 turns and 2 sides — up to ~240 columns scanned per row.

Deliberately NOT reusing ArenaIdCache's cache instance, and NOT wired
into ReplayMetricScanner/ReplayMetric's Protocol: each metric that
needs cast events constructs its OWN CastEventScanner and
calls it itself, inline, inside its own accumulate() — not a
driver-level, declare-then-resolve mechanism. This means each active
turn-scanning metric pays its own resolution (and cache) cost, redundant
across metrics if several are active in one scan — an accepted,
documented tradeoff, not an oversight; revisit centralizing only if
this proves to actually matter in practice.

CastEventScanner's cache-first per-cell resolution logic is
structurally identical to ArenaIdCache._resolve_cell() (same
CardBinder.get_by_alias() call, same cache-then-record-misses shape) —
duplicated here rather than shared, because ArenaIdCache is out of
scope for this change (see this container's own change history) and
because the two caches' STATE must stay independent anyway (each
metric constructs its own CastEventScanner instance; there is no
shared cache to share the logic around). A future cleanup could
extract the common "cache-first Arena ID -> nocab_uuid" primitive both
this and ArenaIdCache build on, once a third such cache exists (rule of
three) or this duplication is otherwise felt as a real cost — not
attempted here.

Because a card can be cast multiple times in one game (bounce/recast),
every cast instance found is its own CastEvent — consumers must NOT
assume at most one CastEvent per (row, card) pair.
"""

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.lookup_cache import LookupCache
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_MAX_TURN = 30  # confirmed via real data this session: every replay_data
# row has user_turn_1..30_*/oppo_turn_1..30_* columns, regardless of
# actual game length (unused turns are empty/zero-filled, not absent)


class Side(str, Enum):
    """Which player a turn-indexed column family belongs to.

    Inputs: none (enum).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    USER = "user"
    OPPO = "oppo"


@dataclass(frozen=True)
class CastEvent:
    """One instance of one card being cast, found in one row.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    side: Side  # which player cast the card
    turn: int  # the turn number embedded in whichever raw column this
    # instance was found in (own-turn or off-turn/flash — see this
    # module's docstring). For an own-turn cast this is unambiguously
    # side's own turn count; for an off-turn cast, the column lives
    # under the OTHER side's turn prefix (e.g.
    # oppo_turn_5_user_instants_sorceries_cast), so this number is
    # literally the other side's turn index, not a verified count of
    # side's own turns taken — treated as "close enough" per this
    # session's relaxed precision tolerance (see
    # average_turns_remaining_post_cast.py's precision note), not
    # claimed as exact.
    nocab_uuid: UUID
    own_turn: bool  # True if found via one of side's own-turn columns
    # (creatures_cast/non_creatures_cast/<side>_instants_sorceries_cast
    # under side's own turn prefix); False if found via the off-turn/
    # flash-speed bucket (<side>_instants_sorceries_cast under the
    # OTHER side's turn prefix). Distinguishes the two cases _cast_columns()
    # conflates into one turn number — a consumer that needs "side's own
    # turn count" unambiguously (e.g. opponent_response_rate.py's
    # same-turn response check) must restrict itself to own_turn=True
    # events; turn is only reliably side's own turn count in that case.


class CastEventScanner:
    """Caches CardBinder lookups for repeated Arena IDs across cast columns.

    Single-consumer to whichever ReplayMetric constructs it — unlike
    ArenaIdCache (owned by ReplayMetricScanner, shared across every
    metric), each metric needing cast events owns its own
    CastEventScanner instance, constructed once in that metric's
    __init__ and reused across every accumulate() call for that
    metric's lifetime (see this module's docstring for why this isn't
    centralized).
    """

    def __init__(self, card_binder: CardBinder, source_game: GameId) -> None:
        """
        Inputs:
            card_binder: registry to resolve Arena IDs against.
            source_game: which game's alias namespace to resolve
                against (GameId.MTG for every 17lands source today).
        Output: none (constructor).
        Side effects: none — no CardBinder queries happen until
            find_cast_events() is called.
        Exceptions: none.
        """

        def lookup(arena_id: str) -> UUID | None:
            card = card_binder.get_by_alias(source_game, DataSource.ARENA, arena_id)
            return card.nocab_uuid if card is not None else None

        self._cache = LookupCache(lookup)

    def find_cast_events(self, chunk: pd.DataFrame) -> pd.Series:
        """Find every cast instance in one chunk, per row.

        For each row, for each Side, for each turn 1..30: reads the 4
        cast-relevant columns for that (side, turn) pair (see this
        module's docstring; _cast_columns() below names them), resolves
        any non-empty cell's Arena IDs via self._resolve_cell()
        (cache-first), and emits one CastEvent per resolved card found
        — NOT deduplicated across cells/turns within a row (unlike
        ArenaIdCache's per-hand dedup): a card cast on turn 3 and again
        on turn 7 is two separate CastEvents, and a card appearing
        twice in the SAME cell (e.g. two creatures cast the same turn,
        both resolving to the same nocab_uuid — confirmed possible from
        real multi-ID cells like "105246|104931") is two separate
        CastEvents too, per this session's "every cast instance is its
        own observation" policy.

        Inputs:
            chunk: one chunk of the raw replay_data CSV — must include
                every <side>_turn_<N>_<family> column this scans (i.e.
                the full raw header, unfiltered).
        Output: a pd.Series, index-aligned with chunk, one
            list[CastEvent] per row (empty list if that row has no
            resolvable casts at all).
        Side effects: mutates self._cache and self._unresolved_ids for
            any Arena ID not already cached (see _resolve_cell()).
        Exceptions: none expected from well-formed input.

        Example:
            >>> scanner = CastEventScanner(card_binder, GameId.MTG)
            >>> events_per_row = scanner.find_cast_events(chunk)
        """
        # Assumes chunk.index has no duplicate labels — a duplicate
        # would collapse two rows into one dict entry here, then the
        # final pd.Series(...) call below would raise on a length
        # mismatch. Holds for pandas.read_csv(chunksize=...)'s normal
        # per-chunk RangeIndex; not otherwise enforced.
        events_by_row: dict[object, list[CastEvent]] = {idx: [] for idx in chunk.index}

        for side in Side:
            for turn in range(1, _MAX_TURN + 1):
                for column, own_turn in _cast_columns(side, turn):
                    cell_series = chunk[column]
                    non_empty = cell_series[cell_series.notna() & (cell_series != "")]
                    for row_index, cell in non_empty.items():
                        for nocab_uuid in self._resolve_cell(cell):
                            events_by_row[row_index].append(
                                CastEvent(
                                    side=side,
                                    turn=turn,
                                    nocab_uuid=nocab_uuid,
                                    own_turn=own_turn,
                                )
                            )

        return pd.Series(list(events_by_row.values()), index=chunk.index)

    def _resolve_cell(self, cell: str) -> list[UUID]:
        """Parse and resolve one pipe-delimited Arena-ID cell, cache-first.

        Private helper — single consumer is find_cast_events(), called
        once per (row, side, turn, family) cell. Identical logic to
        ArenaIdCache._resolve_cell() (see arena_id_cache.py) — splits
        cell on "|", resolves each ID via
        card_binder.get_by_alias(source_game, DataSource.ARENA, id),
        cache-first, records misses rather than raising. UNLIKE
        ArenaIdCache._resolve_cell(), the returned list is NOT
        deduplicated — find_cast_events() needs to know how many times
        a card was cast in one cell, not just whether it was cast at
        all (see find_cast_events()'s docstring).

        Inputs:
            cell: one raw cell value — normally a pipe-delimited string
                of Arena numeric card IDs, or an empty string / NaN.
                Also handles the case where pandas has inferred this
                particular chunk's column as float64 instead of
                string/object (happens when every non-null cell in
                this chunk's slice of the column holds a single,
                un-piped numeric ID, e.g. 104936.0 instead of
                "104936" — confirmed real for sparse cast columns,
                unlike arena_id_cache.py's always-multi-ID hand
                columns) — such a value is coerced via str(int(cell))
                before parsing.
        Output: list of resolved nocab_uuids, one per resolvable ID in
            cell, in cell's own order, WITH duplicates preserved. An
            empty/NaN cell yields [].
        Side effects: mutates the internal cache (always, per distinct
            ID not already cached).
        Exceptions: none expected from well-formed input.
        """
        if pd.isna(cell) or cell == "":
            return []
        if not isinstance(cell, str):
            cell = str(int(cell))

        resolved: list[UUID] = []
        for arena_id in cell.split("|"):
            nocab_uuid = self._cache.get(arena_id)
            if nocab_uuid is not None:
                resolved.append(nocab_uuid)

        return resolved

    @property
    def unresolved_ids(self) -> frozenset[str]:
        """Every distinct Arena ID this scanner has failed to resolve.

        Inputs: none (uses internal state).
        Output: every distinct Arena ID string find_cast_events() has
            been unable to resolve so far, across every call made to
            this instance.
        Side effects: none.
        Exceptions: none.
        """
        return self._cache.unresolved_keys


def _cast_columns(side: Side, turn: int) -> list[tuple[str, bool]]:
    """Name the 4 raw columns that can carry a cast by side on turn.

    Private helper — single consumer is CastEventScanner.find_cast_events().
    Returns, in a fixed order, (column name, own_turn) pairs:
    <side>_turn_<turn>_creatures_cast (own_turn=True),
    <side>_turn_<turn>_non_creatures_cast (own_turn=True),
    <side>_turn_<turn>_<side>_instants_sorceries_cast (own_turn=True), and
    <other side>_turn_<turn>_<side>_instants_sorceries_cast (own_turn=False
    — e.g. for side=USER, this is oppo_turn_<turn>_user_instants_sorceries_cast,
    confirmed real this session). own_turn distinguishes which of the two
    turn-number interpretations applies — see CastEvent.own_turn's
    docstring.

    Inputs:
        side: which player's casts to name columns for.
        turn: that player's own turn number (1..30).
    Output: the 4 (column name, own_turn) pairs, as described above.
    Side effects: none.
    Exceptions: none.
    """
    other = _other_side(side)
    return [
        (f"{side.value}_turn_{turn}_creatures_cast", True),
        (f"{side.value}_turn_{turn}_non_creatures_cast", True),
        (f"{side.value}_turn_{turn}_{side.value}_instants_sorceries_cast", True),
        (f"{other.value}_turn_{turn}_{side.value}_instants_sorceries_cast", False),
    ]


def _other_side(side: Side) -> Side:
    """The opposing Side.

    Private helper — consumed by _cast_columns() and by this module's
    own off_turn_response_column().

    Inputs:
        side: one Side.
    Output: the other Side.
    Side effects: none.
    Exceptions: none.
    """
    return Side.OPPO if side is Side.USER else Side.USER


def off_turn_response_column(caster_side: Side, turn: int) -> str:
    """Name the column recording the OPPOSING side's off-turn cast during
    caster_side's own turn.

    Public accessor — the one column-naming detail this module exposes
    for a consumer that needs to look up a raw cell directly (e.g.
    opponent_response_rate.py checking "did the opposing side respond
    during this cast's own turn") rather than going through
    find_cast_events()'s parsed CastEvent objects. Derived from the
    same _other_side() helper _cast_columns() itself uses — this is
    exactly _cast_columns(caster_side, turn)'s 4th entry's column name,
    named directly rather than re-deriving it from a full
    _cast_columns() call and picking out the one entry a caller needs.
    Added so a consumer never has to reconstruct this format string
    itself (see this module's history — an earlier version of
    opponent_response_rate.py did exactly that, independently of
    _cast_columns(), which would silently drift out of sync with any
    future change here).

    Inputs:
        caster_side: the side whose own turn this is (i.e. the side
            that cast something on-turn; the response being looked up
            is the OTHER side's off-turn cast during this side's turn).
        turn: caster_side's own turn number.
    Output: the raw column name to look up on a replay_data chunk,
        e.g. off_turn_response_column(Side.USER, 5) ->
        "user_turn_5_oppo_instants_sorceries_cast".
    Side effects: none.
    Exceptions: none.

    Example:
        >>> off_turn_response_column(Side.USER, 5)
        'user_turn_5_oppo_instants_sorceries_cast'
    """
    other = _other_side(caster_side)
    return f"{caster_side.value}_turn_{turn}_{other.value}_instants_sorceries_cast"
