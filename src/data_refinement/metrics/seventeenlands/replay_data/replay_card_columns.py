"""Header-derived card index for one 17lands replay_data CSV.

The one genuinely new columns-index shape in this seventeenlands
family (see plans/replay_data_metrics.md's Component overview):
replay_data exposes cards through TWO independent mechanisms, not one.

1. Name-suffixed header columns - deck_<name>/sideboard_<name> ONLY
   (replay_data has no opening_hand_<name> family the way game_data
   does - opening_hand is a single Arena-ID-list column instead, see
   below). Parsed once at construction, via the exact same
   _match_uuid() policy game_data.game_card_columns.GameCardColumns
   and draft_data.pack_pool_columns.DraftCardColumns already
   implement independently for their own sources - a third,
   deliberate copy, already logged in
   src/data_refinement/metrics/TODO.md as a known, intentional
   duplication, not a fresh decision made here.
2. Arena-ID pipe-delimited cells - every per-turn event column
   (creatures_cast, creatures_attacked, ...), plus
   opening_hand/candidate_hand_N/eot_{side}_*_in_play. These can't be
   matched at header-parse time (the ids inside a cell vary row to
   row) - matching happens per row, per cell, through a separate
   cache keyed by the Arena id string itself via
   uuid_for_arena_id()/arena_uuids().

DTYPE TRAP (see plans/replay_data_metrics.md's "What this session
re-verified" section for the full verification): a per-turn Arena-ID
column where every populated cell in a pandas.read_csv chunk happens
to hold exactly one id (no "|" ever needed) is inferred as float64,
not string - e.g. row["user_turn_1_creatures_cast"] can arrive as the
Python float 104936.0, never the string "104936". Meanwhile
ScryfallCardIngestionStage._extract_aliases() registers each Arena
alias as str(row["arena_id"]) on a parsed JSON int - i.e. the ledger's
stored key is "104936", never "104936.0". arena_uuids()/
uuid_for_arena_id() MUST normalize every token (whether it arrives as
a bare float/int or as one "|"-split string piece) via
str(int(float(token))) before querying card_binder - this is a
correctness requirement, not a formatting nicety, and must not be
dropped during implementation.
"""

import re
from typing import ClassVar, Iterable, Literal
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_DECK_PREFIX = "deck_"
_SIDEBOARD_PREFIX = "sideboard_"
_COLUMN_PREFIXES = (_DECK_PREFIX, _SIDEBOARD_PREFIX)

_TURN_COLUMN_PATTERN = re.compile(r"^(user|oppo)_turn_(\d+)_")


class ReplayCardColumns:
    """Every deck_/sideboard_ column of one replay_data CSV matched to
    nocab_uuids, plus a separate Arena-ID cache for every per-turn
    event cell this same scan encounters, plus this CSV's own
    per-actor turn-number range.

    Single-consumer-per-metric: each metric builds its own instance
    (via from_header()) at construction time, from the same
    (card_binder, header, source_game) it was itself constructed with.
    """

    # Every metric in this container that loops both half-turn actors
    # shares this same tuple - centralized here (PRINCIPLES.md section
    # 2) rather than re-typed as a private constant in each metric
    # file.
    ACTORS: ClassVar[tuple[Literal["user", "oppo"], ...]] = ("user", "oppo")

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
        self._user_turn_numbers: list[int] = []
        self._oppo_turn_numbers: list[int] = []

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
            one card (see unmatched_names for the rest), and whose
            user_turn_numbers/oppo_turn_numbers reflect every turn
            number actually present in header for that actor.
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

        # Derive this CSV's own per-actor turn-number range from the
        # header, rather than assuming a fixed max turn.
        result._discover_turn_numbers(header)

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
            matched (see _match_uuid()'s docstring for the exact
            policy).
        Side effects: on a name not already cached, queries
            self._card_binder (via _match_uuid()) and stores the
            result (even if None) for every later call with the same
            name.
        Exceptions: none expected.

        Example:
            >>> replay_columns.uuid_for_name("Lightning Bolt")
        """
        if name in self._name_cache:
            return self._name_cache[name]

        card_uuid = self._match_uuid(name)
        self._name_cache[name] = card_uuid
        return card_uuid

    def _match_uuid(self, name: str) -> UUID | None:
        """The uncached name -> nocab_uuid matching policy itself.

        Private helper - single consumer is uuid_for_name(). The same
        17lands-wide policy DraftCardColumns._match_uuid()/
        GameCardColumns._match_uuid() already implement for their own
        sources, re-typed here rather than shared via inheritance or
        import - see src/data_refinement/metrics/TODO.md.

        Inputs:
            name: a bare card name to match.
        Output: the matching nocab_uuid if
            card_binder.get_by_name(source_game, name) returns exactly
            one card; else the matching nocab_uuid if
            card_binder.get_by_name_regex(source_game,
            f"^{re.escape(name)}( //.*)?$") returns exactly one card;
            else None. Ambiguity (2+ matches from either method) is
            never guessed at - treated identically to no match.
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

    def uuid_for_arena_id(self, arena_id: str) -> UUID | None:
        """Look up an already-normalized Arena id string, caching the
        result.

        Inputs:
            arena_id: an Arena id, already normalized to the ledger's
                own str(int(...)) convention (e.g. "104936", never
                "104936.0") - see module docstring's "DTYPE TRAP".
                Callers with a raw cell value should go through
                arena_uuids() instead, which performs that
                normalization.
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

    def arena_uuids(self, cell: float | int | str | None) -> list[UUID]:
        """Match one per-turn Arena-ID cell into every card it names.

        Inputs:
            cell: one raw cell value from a per-turn event column (or
                opening_hand/candidate_hand_N/eot_*_in_play) - NaN/None
                when empty, a bare float/int when pandas inferred a
                single-id column as numeric, or a "|"-delimited string
                when 2+ ids ever co-occur in that column. See module
                docstring's "DTYPE TRAP" - this method is where that
                trap is closed, not the caller's concern.
        Output: every card_uuid this cell names, matched via
            uuid_for_arena_id() after normalizing each token (whole
            cell, or each "|"-split piece) via str(int(float(token))).
            Empty list if cell is NaN/None/empty, or if every token was
            unmatched (see unmatched_arena_ids).
        Side effects: same caching side effect as uuid_for_arena_id(),
            once per distinct token encountered.
        Exceptions: none expected for a well-formed cell.

        Example:
            >>> replay_columns.arena_uuids("105091|104894")
        """
        if cell is None or pd.isna(cell):
            return []

        cell_str = cell if isinstance(cell, str) else str(cell)

        card_uuids = []
        for token in cell_str.split("|"):
            normalized = str(int(float(token)))
            card_uuid = self.uuid_for_arena_id(normalized)
            if card_uuid is not None:
                card_uuids.append(card_uuid)
        return card_uuids

    def present_uuids(self, row: dict, columns: list[tuple[str, UUID]]) -> list[UUID]:
        """Every matched card from `columns` whose count is > 0 on this
        row.

        Same contract as GameCardColumns.present_uuids() - shared by
        every deck_columns/sideboard_columns consumer in this
        container, since both remain per-row copy COUNTS (deck_<name>
        sums to 40), not per-copy list entries.

        Inputs:
            row: one replay_data CSV row, dict-like.
            columns: deck_columns/sideboard_columns (or any
                same-shaped list).
        Output: every card_uuid from columns whose row[column_name] is
            truthy (nonzero) - order matches columns' own order.
        Side effects: none.
        Exceptions: raises KeyError if a column in columns is missing
            from row.

        Example:
            >>> replay_columns.present_uuids(row, replay_columns.deck_columns)
        """
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

    @property
    def user_turn_numbers(self) -> list[int]:
        """Every turn number N for which at least one
        "user_turn_N_<field>" column exists in this CSV's header,
        sorted ascending.

        Derived once at construction (_discover_turn_numbers()) -
        never a hardcoded range, the same "derive table size at scan
        time" precedent draft_data's pack/table size already
        established.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._user_turn_numbers

    @property
    def oppo_turn_numbers(self) -> list[int]:
        """Every turn number N for which at least one
        "oppo_turn_N_<field>" column exists in this CSV's header,
        sorted ascending. See user_turn_numbers.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._oppo_turn_numbers

    def _discover_turn_numbers(self, header: Iterable[str]) -> None:
        """Populate self._user_turn_numbers/_oppo_turn_numbers by
        regex-scanning header.

        Private helper - single consumer is from_header().

        Inputs:
            header: this CSV's column names.
        Output: none.
        Side effects: sets self._user_turn_numbers/
            _oppo_turn_numbers from every "^(user|oppo)_turn_(\\d+)_"
            match in header, each sorted ascending, deduplicated.
        Exceptions: none expected.
        """
        user_turns = set()
        oppo_turns = set()
        for column in header:
            match = _TURN_COLUMN_PATTERN.match(column)
            if match is None:
                continue
            actor, turn = match.group(1), int(match.group(2))
            if actor == "user":
                user_turns.add(turn)
            else:
                oppo_turns.add(turn)

        self._user_turn_numbers = sorted(user_turns)
        self._oppo_turn_numbers = sorted(oppo_turns)

    @staticmethod
    def turn_column(actor: Literal["user", "oppo"], turn: int, field: str) -> str:
        """Build the literal header column name for one (actor, turn,
        field) triple.

        Centralizes the f"{actor}_turn_{turn}_{field}" format string -
        nearly every metric in this container loops turns and needs
        this exact literal (PRINCIPLES.md section 2 dedup), so it lives
        here rather than being re-typed per metric.

        Inputs:
            actor: which half-turn - "user" or "oppo".
            turn: that actor's own turn-number counter (see
                user_turn_numbers/oppo_turn_numbers - these are
                independent per-actor counters, not a shared
                elapsed-turn index).
            field: the field name suffix (e.g. "creatures_cast",
                "eot_user_life").
        Output: the literal column name, e.g.
            turn_column("user", 3, "creatures_cast") ->
            "user_turn_3_creatures_cast".
        Side effects: none.
        Exceptions: none.

        Example:
            >>> ReplayCardColumns.turn_column("oppo", 5, "creatures_attacked")
            'oppo_turn_5_creatures_attacked'
        """
        return f"{actor}_turn_{turn}_{field}"

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
