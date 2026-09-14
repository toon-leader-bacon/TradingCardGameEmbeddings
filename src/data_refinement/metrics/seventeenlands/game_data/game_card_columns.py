"""Header-derived card-name index for one 17lands game_data CSV.

Built once per metric instance (each metric constructs its own, from
its own constructor's (card_binder, header, source_game) - see
game_card_average_metric.py's module docstring for why this is a
per-metric-cheap, header-sized cost rather than a shared per-scan
object). Every opening_hand_<name>/drawn_<name>/tutored_<name>/
deck_<name>/sideboard_<name> column is parsed and its <name> suffix
matched against card_binder directly (no separate lookup/cache class in
between - see _match_uuid()'s docstring for the matching policy, the
same one draft_data/pack_pool_columns.py's DraftCardColumns._match_uuid()
already implements for that sibling source. This is one 17lands-wide
card-name-matching *policy*, deliberately re-typed here rather than
imported/subclassed from draft_data - see
plans/game_data_metrics.md's "Existing contracts this plan depends on"
section for why this duplication is intentional, not an oversight).

Unlike DraftCardColumns, game_data has no per-row cell value analogous
to draft_data's `pick` column - every name this class ever looks up
comes from a header column suffix, never a row's own value.
uuid_for_name() is kept public anyway, for the same testability and
structural consistency reasons DraftCardColumns keeps it public.
"""

import re
from typing import Iterable
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
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

    Single-consumer-per-metric: each metric builds its own instance (via
    from_header()) at construction time, from the same (card_binder,
    header, source_game) it was itself constructed with - see
    game_card_average_metric.py's __init__.
    """

    def __init__(self, card_binder: CardBinder, source_game: GameId) -> None:
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

    @staticmethod
    def from_header(
        columns: Iterable[str], card_binder: CardBinder, source_game: GameId
    ) -> "GameCardColumns":
        """Parse and match one CSV's header into a GameCardColumns.

        Factory Method (PATTERNS.md) - the only sanctioned way to build
        a populated instance.

        Inputs:
            columns: a game_data CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - only
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
            >>> header = pd.read_csv(raw_csv_path, nrows=0).columns
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
            matched (see _match_uuid()'s docstring for the exact
            policy).
        Side effects: on a name not already cached, queries
            self._card_binder (via _match_uuid()) and stores the result
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
        card_uuid = self._match_uuid(name)
        self._cache[name] = card_uuid
        return card_uuid

    def _match_uuid(self, name: str) -> UUID | None:
        """The uncached name -> nocab_uuid matching policy itself.

        Private helper - single consumer is uuid_for_name(). The same
        17lands-wide policy draft_data/pack_pool_columns.py's
        DraftCardColumns._match_uuid() implements for that sibling
        source, re-typed here rather than shared via inheritance or
        import - see plans/game_data_metrics.md's "Existing contracts"
        section for why this duplication is intentional.

        Inputs:
            name: a bare card name to match.
        Output: the matching nocab_uuid if
            card_binder.get_by_name(source_game, name) returns exactly
            one card; else the matching nocab_uuid if
            card_binder.get_by_name_regex(source_game,
            f"^{re.escape(name)}( //.*)?$") (name as a split/MDFC card's
            front face) returns exactly one card; else None. Ambiguity
            (2+ matches from either method) is never guessed at -
            treated identically to no match.
        Side effects: none - read-only card_binder queries.
        Exceptions: none expected.
        """
        exact_matches = self._card_binder.get_by_name(self._source_game, name)
        if len(exact_matches) == 1:
            return exact_matches[0].nocab_uuid

        front_face_matches = self._card_binder.get_by_name_regex(
            self._source_game, f"^{re.escape(name)}( //.*)?$"
        )
        if len(front_face_matches) == 1:
            return front_face_matches[0].nocab_uuid

        return None

    def present_uuids(self, row: dict, columns: list[tuple[str, UUID]]) -> list[UUID]:
        """Every matched card from `columns` whose count is > 0 on this
        row.

        Shared by every metric in this container - called with any of
        opening_hand_columns/drawn_columns/tutored_columns/
        deck_columns/sideboard_columns (or any same-shaped list), so
        the "which columns are actually present on this row" filtering
        logic exists exactly once. Every game_data column here is a
        per-game copy COUNT, not a per-copy list entry (deck_<name>
        sums to 40, opening_hand_<name> to 7) - this method samples on
        presence (count > 0) exactly once per qualifying card, never
        once per copy - see plans/game_data_metrics.md's "Data facts"
        section for why this differs from sts_gg's per-copy tallying.

        Inputs:
            row: one game_data CSV row, dict-like (column name -> cell
                value) - see ../scanner.py's module docstring for where
                this comes from.
            columns: one of the *_columns properties (or any
                same-shaped list) to filter against row.
        Output: every card_uuid from columns whose row[column_name] is
            truthy (nonzero) - order matches columns' own order.
        Side effects: none.
        Exceptions: raises KeyError if a column in columns is missing
            from row.

        Example:
            >>> game_columns.present_uuids(row, game_columns.deck_columns)
        """
        # pd.notna() treats a NaN/missing cell as absent regardless of
        # `bool(float("nan"))` being True in plain Python; the `and`
        # then still requires an actually-truthy (nonzero) count.
        return [
            card_uuid
            for column_name, card_uuid in columns
            if pd.notna(row[column_name]) and row[column_name]
        ]

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
