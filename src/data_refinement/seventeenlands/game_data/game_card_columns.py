"""Header-derived card-name index for one 17lands game_data CSV.

Built once per CSV by GameDataChunkParser.from_header()
(game_data_chunk_parser.py), which regroups its matched columns into
per-zone count matrices. Every opening_hand_<name>/drawn_<name>/tutored_<name>/
deck_<name>/sideboard_<name> column is parsed and its <name> suffix
matched against card_binder with
card_lookup.uuid_for_name_or_front_face(), the 17lands-wide matching
policy every sibling columns index uses (no separate lookup/cache class
in between).

Unlike DraftCardColumns, game_data has no per-row cell value analogous
to draft_data's `pick` column - every name this class ever looks up
comes from a header column suffix, never a row's own value.
uuid_for_name() is kept public anyway, for the same testability and
structural consistency reasons DraftCardColumns keeps it public.
"""

from typing import Iterable
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import (
    CardLookup,
    uuid_for_name_or_front_face,
)
from src.schema.game_id import GameId

_OPENING_HAND_PREFIX = "opening_hand_"
_DRAWN_PREFIX = "drawn_"
_TUTORED_PREFIX = "tutored_"
_DECK_PREFIX = "deck_"
_SIDEBOARD_PREFIX = "sideboard_"

_COLUMN_PREFIXES = (
    _OPENING_HAND_PREFIX,
    _DRAWN_PREFIX,
    _TUTORED_PREFIX,
    _DECK_PREFIX,
    _SIDEBOARD_PREFIX,
)


class GameCardColumns:
    """Every opening_hand_/drawn_/tutored_/deck_/sideboard_ column of
    one game_data CSV, matched to nocab_uuids, plus a cache for any
    other bare name this same scan encounters.

    Single consumer: GameDataChunkParser.from_header() builds one per
    CSV.
    """

    def __init__(self, card_binder: CardLookup, source_game: GameId) -> None:
        """
        Inputs:
            card_binder: registry to look up card names against -
                assumed already fully populated for source_game.
            source_game: which game's cards to look up against.
        Output: none (constructor).
        Side effects: none - no card_binder queries happen until
            from_header() or uuid_for_name() is called.
        Exceptions: none.
        """
        self._card_binder = card_binder
        self._source_game = source_game
        self._cache: dict[str, UUID | None] = {}
        self._opening_hand_columns: list[tuple[str, UUID]] = []
        self._drawn_columns: list[tuple[str, UUID]] = []
        self._tutored_columns: list[tuple[str, UUID]] = []
        self._deck_columns: list[tuple[str, UUID]] = []
        self._sideboard_columns: list[tuple[str, UUID]] = []
        self._unmatched_deck_columns: list[str] = []

    @staticmethod
    def from_header(
        columns: Iterable[str], card_binder: CardLookup, source_game: GameId
    ) -> "GameCardColumns":
        """Parse and match one CSV's header into a GameCardColumns.

        Factory Method (PATTERNS.md) - the only sanctioned way to build
        a populated instance.

        Inputs:
            columns: a game_data CSV's column names (e.g.
                csv_header.read_csv_header(path)) - only
                "opening_hand_<name>"/"drawn_<name>"/"tutored_<name>"/
                "deck_<name>"/"sideboard_<name>"-prefixed entries are
                inspected, every other column name (expansion,
                event_type, rank, won, etc.) is ignored.
            card_binder: registry to look up each distinct <name>
                against - assumed already fully populated for
                source_game.
            source_game: which game's cards header names are looked up
                against.
        Output: a GameCardColumns whose opening_hand_columns/
            drawn_columns/tutored_columns/deck_columns/
            sideboard_columns cover every column that matched exactly
            one card; every column whose <name> matched none is simply
            absent from all five (see unmatched_names to find out
            which).
        Side effects: none beyond constructing the returned instance (no
            I/O - card_binder is assumed already loaded).
        Exceptions: none expected.

        Example:
            >>> header = read_csv_header(raw_csv_path)
            >>> GameCardColumns.from_header(header, card_binder, GameId.MTG)
        """
        game_columns = GameCardColumns(card_binder, source_game)

        # Sort each header column into one of the five prefixes (or
        # ignored), and match its <name> suffix through game_columns'
        # own cache.
        for column in columns:
            parsed = GameCardColumns._card_name_for_column(column)
            if parsed is None:
                continue
            prefix, card_name = parsed

            card_uuid = game_columns.uuid_for_name(card_name)
            if card_uuid is None:
                if prefix == _DECK_PREFIX:
                    game_columns._unmatched_deck_columns.append(column)
                continue

            game_columns._add_matched_column(prefix, column, card_uuid)

        return game_columns

    def _add_matched_column(self, prefix: str, column: str, card_uuid: UUID) -> None:
        """File one already-matched (column, card_uuid) pair into the
        *_columns list its prefix belongs to.

        Private helper - single consumer is from_header(). Exists so
        the five-way "which list does this prefix belong to" dispatch
        is a single, visible, named step rather than inline branching
        in from_header() itself.

        Inputs:
            prefix: one of _OPENING_HAND_PREFIX/_DRAWN_PREFIX/
                _TUTORED_PREFIX/_DECK_PREFIX/_SIDEBOARD_PREFIX, as
                returned by _card_name_for_column().
            column: the full header column name (e.g.
                "deck_Owlbear").
            card_uuid: column's already-matched nocab_uuid.
        Output: none.
        Side effects: appends (column, card_uuid) to whichever of
            self._opening_hand_columns/_drawn_columns/_tutored_columns/
            _deck_columns/_sideboard_columns corresponds to prefix.
        Exceptions: none expected (prefix is always one of the five
            module-level constants, since it only ever comes from
            _card_name_for_column()'s own return value).
        """
        column_lists = {
            _OPENING_HAND_PREFIX: self._opening_hand_columns,
            _DRAWN_PREFIX: self._drawn_columns,
            _TUTORED_PREFIX: self._tutored_columns,
            _DECK_PREFIX: self._deck_columns,
            _SIDEBOARD_PREFIX: self._sideboard_columns,
        }
        column_lists[prefix].append((column, card_uuid))

    @property
    def opening_hand_columns(self) -> list[tuple[str, UUID]]:
        """Every matched "opening_hand_<name>" column, as (column name,
        nocab_uuid) pairs, in the header's own column order.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._opening_hand_columns

    @property
    def drawn_columns(self) -> list[tuple[str, UUID]]:
        """Every matched "drawn_<name>" column, as (column name,
        nocab_uuid) pairs, in the header's own column order.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._drawn_columns

    @property
    def tutored_columns(self) -> list[tuple[str, UUID]]:
        """Every matched "tutored_<name>" column, as (column name,
        nocab_uuid) pairs, in the header's own column order.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._tutored_columns

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
    def unmatched_deck_columns(self) -> list[str]:
        """Every "deck_<name>" column whose <name> matched no card, in the
        header's own column order. Absent from deck_columns (no metric
        tallies them), but still part of the deck's identity: see
        GameDataChunkParser's deck identity columns.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._unmatched_deck_columns

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
        """Look up a bare card name (a header column's <name> suffix),
        caching the result.

        Inputs:
            name: a bare card name - NOT an
                "opening_hand_"/"drawn_"/"tutored_"/"deck_"/
                "sideboard_"-prefixed column name (use the *_columns
                properties for those).
        Output: the matching nocab_uuid, or None if no single card
            matched (see card_lookup.uuid_for_name_or_front_face()).
        Side effects: on a name not already cached, queries
            self._card_binder (via uuid_for_name_or_front_face()) and stores the result
            (even if None) for every later call with the same name.
        Exceptions: none expected.

        Example:
            >>> game_columns.uuid_for_name("Lightning Bolt")
        """
        # Fast path: already looked this name up before (hit or miss).
        if name in self._cache:
            return self._cache[name]

        # Slow path: query card_binder once, cache whatever comes back
        # (including None), so a repeat call never re-queries.
        card_uuid = uuid_for_name_or_front_face(
            self._card_binder, self._source_game, name
        )
        self._cache[name] = card_uuid
        return card_uuid

    @property
    def unmatched_names(self) -> list[str]:
        """Every distinct name uuid_for_name() has returned None for so
        far.

        Inputs: none (uses internal state).
        Output: every key of self._cache whose value is None. Order not
            guaranteed.
        Side effects: none.
        Exceptions: none.
        """
        return [name for name, card_uuid in self._cache.items() if card_uuid is None]

    @staticmethod
    def _card_name_for_column(column: str) -> tuple[str, str] | None:
        """Split one header column into (prefix, card_name), or None if
        it isn't an opening_hand_/drawn_/tutored_/deck_/sideboard_
        -prefixed column.

        Private helper - single consumer is from_header(). The five
        prefixes are mutually exclusive string prefixes (none is a
        prefix of another), so a single ordered startswith dispatch
        classifies every column unambiguously.

        Inputs:
            column: one raw CSV header column name.
        Output: (one of _OPENING_HAND_PREFIX/_DRAWN_PREFIX/
            _TUTORED_PREFIX/_DECK_PREFIX/_SIDEBOARD_PREFIX, the
            remaining suffix as a bare card name), or None if column
            matches none of the five.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> GameCardColumns._card_name_for_column("deck_Owlbear")
            ('deck_', 'Owlbear')
        """
        for prefix in _COLUMN_PREFIXES:
            if column.startswith(prefix):
                return prefix, column[len(prefix) :]
        return None
