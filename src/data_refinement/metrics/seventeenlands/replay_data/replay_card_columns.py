"""Header-derived card index for one 17lands replay_data CSV (see
replay_data/README.md). Built once per CSV by its ReplayDataChunkParser.

replay_data names cards two ways:

1. Name-suffixed header columns - deck_<name>/sideboard_<name> only.
   Parsed once and matched with card_lookup.uuid_for_name_or_front_face(),
   the policy GameCardColumns and DraftCardColumns use.
2. Arena ids inside the per-half-turn cells (creatures_cast, ...). The
   ids vary row to row, so they are matched by the parser, once per
   distinct id, through uuid_for_arena_id()'s cache. An id must be
   normalized to the ledger's str(int(...)) form first
   (ScryfallCardIngestionStage registers Arena aliases as "104936",
   never "104936.0").
"""

from typing import Iterable
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import uuid_for_name_or_front_face
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_DECK_PREFIX = "deck_"
_SIDEBOARD_PREFIX = "sideboard_"
_COLUMN_PREFIXES = (_DECK_PREFIX, _SIDEBOARD_PREFIX)


class ReplayCardColumns:
    """Every deck_/sideboard_ column of one replay_data CSV matched to
    nocab_uuids, plus a separate Arena-id cache.

    Single consumer: the CSV's ReplayDataChunkParser builds one (via
    from_header()).
    """

    def __init__(self, card_binder: CardBinder, source_game: GameId) -> None:
        """
        Inputs:
            card_binder: registry to look up card names/Arena ids
                against - assumed already fully populated for
                source_game.
            source_game: which game's cards to look up against.
        Output: none (constructor).
        Side effects: none - no card_binder queries happen until
            from_header()/uuid_for_name()/uuid_for_arena_id() is
            called.
        Exceptions: none.
        """
        self._card_binder = card_binder
        self._source_game = source_game
        self._name_cache: dict[str, UUID | None] = {}
        self._arena_id_cache: dict[str, UUID | None] = {}
        self._deck_columns: list[tuple[str, UUID]] = []
        self._sideboard_columns: list[tuple[str, UUID]] = []

    @staticmethod
    def from_header(
        header: Iterable[str], card_binder: CardBinder, source_game: GameId
    ) -> "ReplayCardColumns":
        """Parse one replay_data CSV's header into a ReplayCardColumns.

        Factory Method (PATTERNS.md) - the only sanctioned way to build
        a populated instance.

        Inputs:
            header: a replay_data CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns).
            card_binder: registry to look up each distinct deck_/
                sideboard_ <name> against - assumed already fully
                populated for source_game.
            source_game: which game's cards header names are looked up
                against.
        Output: a ReplayCardColumns whose deck_columns/
            sideboard_columns cover every column that matched exactly
            one card (see unmatched_names for the rest).
        Side effects: none beyond constructing the returned instance
            (no I/O - card_binder is assumed already loaded).
        Exceptions: none expected.

        Example:
            >>> header = pd.read_csv(raw_csv_path, nrows=0).columns
            >>> ReplayCardColumns.from_header(header, card_binder, GameId.MTG)
        """
        header = list(header)
        result = ReplayCardColumns(card_binder, source_game)

        # Sort each header column into deck_/sideboard_ (or ignore it),
        # matching its <name> suffix through result's own name cache.
        for column in header:
            parsed = ReplayCardColumns._card_name_for_column(column)
            if parsed is None:
                continue
            prefix, card_name = parsed

            card_uuid = result.uuid_for_name(card_name)
            if card_uuid is None:
                continue

            result._add_matched_column(prefix, column, card_uuid)

        return result

    def _add_matched_column(self, prefix: str, column: str, card_uuid: UUID) -> None:
        """File one already-matched (column, card_uuid) pair into the
        deck_columns/sideboard_columns list its prefix belongs to.

        Private helper - single consumer is from_header().

        Inputs:
            prefix: one of _DECK_PREFIX/_SIDEBOARD_PREFIX, as returned
                by _card_name_for_column().
            column: the full header column name (e.g. "deck_Owlbear").
            card_uuid: column's already-matched nocab_uuid.
        Output: none.
        Side effects: appends (column, card_uuid) to whichever of
            self._deck_columns/_sideboard_columns corresponds to
            prefix.
        Exceptions: none expected (prefix is always one of the two
            module-level constants, since it only ever comes from
            _card_name_for_column()'s own return value).
        """
        column_lists = {
            _DECK_PREFIX: self._deck_columns,
            _SIDEBOARD_PREFIX: self._sideboard_columns,
        }
        column_lists[prefix].append((column, card_uuid))

    @property
    def deck_columns(self) -> list[tuple[str, UUID]]:
        """Every matched "deck_<name>" column, as (column name,
        nocab_uuid) pairs, in the header's own column order.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._deck_columns

    @property
    def sideboard_columns(self) -> list[tuple[str, UUID]]:
        """Every matched "sideboard_<name>" column, as (column name,
        nocab_uuid) pairs, in the header's own column order.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._sideboard_columns

    def uuid_for_name(self, name: str) -> UUID | None:
        """Look up a bare card name (a deck_/sideboard_ column's
        <name> suffix), caching the result.

        Inputs:
            name: a bare card name - NOT a "deck_"/"sideboard_"
                -prefixed column name (use deck_columns/
                sideboard_columns for those).
        Output: the matching nocab_uuid, or None if no single card
            matched (see card_lookup.uuid_for_name_or_front_face()).
        Side effects: on a name not already cached, queries
            self._card_binder (via uuid_for_name_or_front_face()) and stores the
            result (even if None) for every later call with the same
            name.
        Exceptions: none expected.

        Example:
            >>> replay_columns.uuid_for_name("Lightning Bolt")
        """
        if name in self._name_cache:
            return self._name_cache[name]

        card_uuid = uuid_for_name_or_front_face(
            self._card_binder, self._source_game, name
        )
        self._name_cache[name] = card_uuid
        return card_uuid

    def uuid_for_arena_id(self, arena_id: str) -> UUID | None:
        """Look up an already-normalized Arena id string, caching the
        result.

        Inputs:
            arena_id: an Arena id, already normalized to the ledger's
                own str(int(...)) convention (e.g. "104936", never
                "104936.0"); the parser normalizes raw tokens first.
        Output: the matching nocab_uuid, or None if
            card_binder.get_by_alias(source_game, DataSource.ARENA,
            arena_id) has no match.
        Side effects: on an arena_id not already cached, queries
            self._card_binder and stores the result (even if None) for
            every later call with the same arena_id. Uses a cache
            separate from uuid_for_name()'s - an Arena id and a bare
            card name are different key spaces.
        Exceptions: none expected.

        Example:
            >>> replay_columns.uuid_for_arena_id("104936")
        """
        if arena_id in self._arena_id_cache:
            return self._arena_id_cache[arena_id]

        card = self._card_binder.get_by_alias(
            self._source_game, DataSource.ARENA, arena_id
        )
        card_uuid = card.nocab_uuid if card is not None else None
        self._arena_id_cache[arena_id] = card_uuid
        return card_uuid

    @property
    def unmatched_names(self) -> list[str]:
        """Every distinct name uuid_for_name() has returned None for so
        far.

        Inputs: none (uses internal state).
        Output: every name-cache key whose value is None. Order not
            guaranteed.
        Side effects: none.
        Exceptions: none.
        """
        return [
            name for name, card_uuid in self._name_cache.items() if card_uuid is None
        ]

    @property
    def unmatched_arena_ids(self) -> list[str]:
        """Every distinct (already-normalized) Arena id
        uuid_for_arena_id() has returned None for so far.

        Inputs: none (uses internal state).
        Output: every Arena-id-cache key whose value is None. Order
            not guaranteed.
        Side effects: none.
        Exceptions: none.
        """
        return [
            arena_id
            for arena_id, card_uuid in self._arena_id_cache.items()
            if card_uuid is None
        ]

    @staticmethod
    def _card_name_for_column(column: str) -> tuple[str, str] | None:
        """Split one header column into (prefix, card_name), or None if
        it isn't a deck_/sideboard_-prefixed column.

        Private helper - single consumer is from_header(). Same
        dispatch shape as GameCardColumns._card_name_for_column(), just
        over this container's own two-prefix tuple.

        Inputs:
            column: one raw CSV header column name.
        Output: (one of _DECK_PREFIX/_SIDEBOARD_PREFIX, the remaining
            suffix as a bare card name), or None if column matches
            neither.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> ReplayCardColumns._card_name_for_column("deck_Owlbear")
            ('deck_', 'Owlbear')
        """
        for prefix in _COLUMN_PREFIXES:
            if column.startswith(prefix):
                return prefix, column[len(prefix) :]
        return None
