"""Header-derived card-name index for one 17lands draft_data CSV.

Built once per metric instance (each metric constructs its own, from
its own constructor's (card_binder, header, source_game) — see
../pack_card_tally_metric.py's module docstring for why this is a
per-metric-cheap, header-sized cost rather than a shared per-scan
object). Every pack_card_<name>/pool_<name> column is parsed and its
<name> suffix matched against card_binder directly (no separate
lookup/cache class in between — see _match_uuid()'s docstring for the
matching policy this recovers from the deleted legacy name_lookup.py,
folded into this class instead of its own file/class, per
plans/draft_data_metrics.md's "Existing contracts" section).

The same name -> nocab_uuid matching also answers the `pick` column's
per-row cell value (see plans/draft_data_metrics.md's "Data facts"
section) - a pick's name is always drawn from the same per-set card
pool as the pack_card_/pool_ column suffixes, so uuid_for_name() serves
both header-driven and value-driven lookups through the same cache.
"""

import re
from typing import Iterable
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.game_id import GameId

_PACK_CARD_PREFIX = "pack_card_"
_POOL_PREFIX = "pool_"


class DraftCardColumns:
    """Every pack_card_/pool_ column of one draft_data CSV, matched to
    nocab_uuids, plus a cache for any other bare name (e.g. a `pick`
    cell value) this same scan encounters.

    Single-consumer-per-metric: each metric builds its own instance (via
    from_header()) at construction time, from the same (card_binder,
    header, source_game) it was itself constructed with - see
    ../pack_card_tally_metric.py's __init__.
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
        self._pack_columns: list[tuple[str, UUID]] = []
        self._pool_columns: list[tuple[str, UUID]] = []

    @staticmethod
    def from_header(
        columns: Iterable[str], card_binder: CardBinder, source_game: GameId
    ) -> "DraftCardColumns":
        """Parse and match one CSV's header into a DraftCardColumns.

        Factory Method (PATTERNS.md) - the only sanctioned way to build
        a populated instance.

        Inputs:
            columns: a draft_data CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - only
                "pack_card_<name>"/"pool_<name>"-prefixed entries are
                inspected, every other column name is ignored.
            card_binder: registry to look up each distinct <name>
                against - assumed already fully populated for
                source_game.
            source_game: which game's cards header/pick names are
                looked up against.
        Output: a DraftCardColumns whose pack_columns/pool_columns
            cover every column that matched exactly one card; every
            column whose <name> matched none is simply absent from both
            lists (see unmatched_names to find out which).
        Side effects: none beyond constructing the returned instance (no
            I/O - card_binder is assumed already loaded).
        Exceptions: none expected.

        Example:
            >>> header = pd.read_csv(raw_csv_path, nrows=0).columns
            >>> DraftCardColumns.from_header(header, card_binder, GameId.MTG)
        """
        draft_columns = DraftCardColumns(card_binder, source_game)

        # Sort each header column into pack_card_/pool_/ignored, and
        # match its <name> suffix through draft_columns' own cache.
        for column in columns:
            parsed = DraftCardColumns._card_name_for_column(column)
            if parsed is None:
                continue
            prefix, card_name = parsed

            card_uuid = draft_columns.uuid_for_name(card_name)
            if card_uuid is None:
                continue

            if prefix == _PACK_CARD_PREFIX:
                draft_columns._pack_columns.append((column, card_uuid))
            else:
                draft_columns._pool_columns.append((column, card_uuid))

        return draft_columns

    @property
    def pack_columns(self) -> list[tuple[str, UUID]]:
        """Every matched "pack_card_<name>" column, as (column name,
        nocab_uuid) pairs, in the header's own column order.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._pack_columns

    @property
    def pool_columns(self) -> list[tuple[str, UUID]]:
        """Every matched "pool_<name>" column, as (column name,
        nocab_uuid) pairs, in the header's own column order.

        Inputs: none.
        Output: see above.
        Side effects: none.
        Exceptions: none.
        """
        return self._pool_columns

    def uuid_for_name(self, name: str) -> UUID | None:
        """Look up a bare card name (e.g. a `pick` cell's value, or a
        header column's <name> suffix), caching the result.

        Inputs:
            name: a bare card name - NOT a "pack_card_"/"pool_"
                prefixed column name (use pack_columns/pool_columns for
                those).
        Output: the matching nocab_uuid, or None if no single card
            matched (see _match_uuid()'s docstring for the exact
            policy).
        Side effects: on a name not already cached, queries
            self._card_binder (via _match_uuid()) and stores the result
            (even if None) for every later call with the same name.
        Exceptions: none expected.

        Example:
            >>> draft_columns.uuid_for_name(row["pick"])
        """
        if name in self._cache:
            return self._cache[name]

        card_uuid = self._match_uuid(name)
        self._cache[name] = card_uuid
        return card_uuid

    def _match_uuid(self, name: str) -> UUID | None:
        """The uncached name -> nocab_uuid matching policy itself.

        Private helper - single consumer is uuid_for_name(). Recovers
        the legacy find_uuid_by_name/PickNameCache policy (deleted in
        commit 1616b6d, recovered via `git show
        1616b6d^:src/data_refinement/metrics/legacy/seventeenlands/name_lookup.py`
        and `git show
        f88a2e0:src/data_refinement/seventeenlands/draft_data_metrics/pick_name_cache.py`)
        directly against card_binder, with no separate lookup/cache
        class in between - see plans/draft_data_metrics.md's "Existing
        contracts this plan builds on" section.

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

        Shared by every metric in this container - called with either
        self.pack_columns (to get a row's pack options) or
        self.pool_columns (to get a row's pool-so-far), so the "which
        columns are actually present on this row" filtering logic
        exists exactly once.

        Inputs:
            row: one draft_data CSV row, dict-like (column name -> cell
                value) - see ../scanner.py's module docstring for where
                this comes from.
            columns: pack_columns or pool_columns (or any same-shaped
                list) to filter against row.
        Output: every card_uuid from columns whose row[column_name] is
            truthy (nonzero) - order matches columns' own order.
        Side effects: none.
        Exceptions: raises KeyError if a column in columns is missing
            from row.

        Example:
            >>> draft_columns.present_uuids(row, draft_columns.pack_columns)
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
        it isn't a pack_card_/pool_-prefixed column.

        Private helper - single consumer is from_header().

        Inputs:
            column: one raw CSV header column name.
        Output: (_PACK_CARD_PREFIX or _POOL_PREFIX, the remaining
            suffix as a bare card name), or None if column matches
            neither prefix.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> DraftCardColumns._card_name_for_column("pack_card_Bruce Banner")
            ('pack_card_', 'Bruce Banner')
        """
        if column.startswith(_PACK_CARD_PREFIX):
            return _PACK_CARD_PREFIX, column[len(_PACK_CARD_PREFIX) :]
        if column.startswith(_POOL_PREFIX):
            return _POOL_PREFIX, column[len(_POOL_PREFIX) :]
        return None
