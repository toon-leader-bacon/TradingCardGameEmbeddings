"""TableDiet sampling: a plan's diet table (src/training/plan.py) turned
into a DropTable of dojo names over the dojos still drawable.

One build (build_dojo_drop_table) decides what is drawable, for both the
sampler and the expected-share report (diet_shares.py), and names the
rows it dropped.
"""

import logging
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.dojos.dojo import Dojo
from src.training.diet.count_weighting import TrainCountCache, alpha_of, count_weight
from src.training.plan import (
    DietRule,
    DietTableRow,
    DojoGroupRow,
    DojoRow,
    TableDiet,
    leaf_dojo_names,
)
from src.utils.drop_table import DropTable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmptiedRow:
    """A DojoGroupRow or SubTableRow dropped for having nothing drawable.

    path: its config-style path, e.g. "table[1].table[0]".
    dojo_names: every dojo under it.
    """

    path: str
    dojo_names: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.path} (dojos: {', '.join(self.dojo_names)})"


@dataclass(frozen=True)
class DojoDropTable:
    """A diet table built over the drawable dojos.

    table: the DropTable of dojo names, or None if nothing is drawable.
    emptied_rows: every row dropped for having nothing drawable, in row
        order (a dropped sub-table's own emptied rows come first).
    """

    table: DropTable[str] | None
    emptied_rows: tuple[EmptiedRow, ...]


class TableDietSampler:
    """Concrete DietSampler (diet_sampler.py; Strategy) for a TableDiet.

    Rebuilds its DropTable only when the active set changes (a dojo
    saturates, reactivates or is quarantined). A row left with nothing
    drawable drops out, its share going to its siblings; each such row is
    logged once per sampler.
    """

    def __init__(self, diet: TableDiet) -> None:
        """
        Inputs: diet (TableDiet).
        Output: none (constructor). Side effects: none. Exceptions: none.
        """
        self._diet = diet
        self._counts = TrainCountCache()
        self._active_names: frozenset[str] | None = None
        self._table: DropTable[str] | None = None
        self._logged_empty_rows: set[EmptiedRow] = set()

    def next_dojo(self, active: Sequence[Dojo], rng: random.Random) -> Dojo:
        """Pull one dojo from the table, restricted to active.

        Inputs: active (non-empty Sequence[Dojo], each in the table),
            rng (random.Random).
        Output: one member of active.
        Side effects: advances rng; on a changed active set, reads TRAIN
            counts (cached) and logs newly emptied rows at WARNING.
        Exceptions: ValueError if active is empty, names a dojo outside the
            table, or no active dojo is drawable.

        Example:
            >>> sampler = TableDietSampler(TableDiet((DojoRow(1, "a"),)))
            >>> sampler.next_dojo([dojo_a], random.Random(0)).name
            'a'
        """
        # Validate inputs
        if not active:
            raise ValueError("no active dojos to sample from")
        by_name = {dojo.name: dojo for dojo in active}

        # Rebuild the table only when the active set changed
        names = frozenset(by_name)
        if names != self._active_names:
            self._table = self._table_for(active)
            self._active_names = names

        # Draw one
        assert self._table is not None  # set above with self._active_names
        return by_name[self._table.pull(rng)]

    def _table_for(self, active: Sequence[Dojo]) -> DropTable[str]:
        """The DropTable over active's dojos, logging rows newly emptied.

        Private helper - single caller is next_dojo().
        Inputs: active. Output: DropTable[str].
        Side effects: reads TRAIN counts; logs.
        Exceptions: ValueError as next_dojo().
        """
        # Validate inputs: the build would silently skip a stranger
        outside = sorted({dojo.name for dojo in active} - set(self._diet.dojo_names))
        if outside:
            raise ValueError(f"active dojos {outside} are not in the diet table")

        # Build over the active dojos' counts
        counts = {dojo.name: self._counts.count(dojo) for dojo in active}
        built = build_dojo_drop_table(self._diet, counts)

        # Log each row the first time it empties
        for row in built.emptied_rows:
            if row not in self._logged_empty_rows:
                self._logged_empty_rows.add(row)
                logger.warning(
                    "diet row %s has no drawable dojo left; its share goes "
                    "to the rows beside it",
                    row,
                )
        if built.table is None:
            # Name the emptied rows, or the dojos when only single rows emptied
            emptied = [str(row) for row in built.emptied_rows] or sorted(counts)
            raise ValueError(
                f"no active dojo is drawable (no TRAIN examples or zero "
                f"weight): {', '.join(emptied)}"
            )
        return built.table


def as_table_diet(rule: DietRule, dojo_names: Sequence[str]) -> TableDiet:
    """rule as a TableDiet: itself, or a flat rule as one DojoGroupRow over
    dojo_names.

    Inputs: rule, dojo_names (non-empty for a flat rule).
    Output: TableDiet. Side effects: none.
    Exceptions: ValueError for a flat rule with no dojos.

    Example:
        >>> as_table_diet(Uniform(), ["a"]).rows
        (DojoGroupRow(weight=1.0, dojo_names=('a',), within=Uniform()),)
    """
    if isinstance(rule, TableDiet):
        return rule
    return TableDiet((DojoGroupRow(1.0, tuple(dojo_names), rule),))


def build_dojo_drop_table(diet: TableDiet, counts: Mapping[str, int]) -> DojoDropTable:
    """diet as a DropTable over its drawable dojos, plus the rows dropped.

    A dojo is drawable when it is in counts with a positive count and its
    own weight (a DojoRow's weight, or count ** alpha in a group) is
    positive. A DojoRow keeps its weight; a DojoGroupRow becomes a
    sub-table weighted count ** alpha_of(within); a SubTableRow becomes a
    sub-table. A row with no drawable dojo, or whose own weight is 0 over
    only zero-weight survivors, drops out.

    Inputs: diet, counts (TRAIN example count per available dojo; a dojo
        not in counts is unavailable, e.g. saturated).
    Output: DojoDropTable.
    Side effects: none. Exceptions: none.

    Example:
        >>> build_dojo_drop_table(TableDiet((DojoRow(1, "a"),)), {"a": 5}).table
        DropTable(entries=(DropTableEntry(weight=1, outcome='a'),))
    """
    emptied: list[EmptiedRow] = []
    table = _drop_table_from_rows(diet.rows, counts, "table", emptied)
    return DojoDropTable(table, tuple(emptied))


def _drop_table_from_rows(
    rows: tuple[DietTableRow, ...],
    counts: Mapping[str, int],
    path: str,
    emptied: list[EmptiedRow],
) -> DropTable[str] | None:
    """rows as a DropTable, or None if nothing in them is drawable; every
    group or sub-table row dropped is appended to emptied.

    Private helper - callers are build_dojo_drop_table() and itself.
    Recursive over SubTableRows: the nesting mirrors the config's own
    shape (a few levels, written by hand), as DropTable.filtered does.
    Inputs: rows, counts, path (the rows' config path, e.g.
        "table[1].table"), emptied (accumulator).
    Output: DropTable[str] | None.
    Side effects: appends to emptied. Exceptions: none.
    """
    entries: list[tuple[float, str | DropTable[str]]] = []

    # Each row becomes an outcome, a sub-table, or nothing
    for index, row in enumerate(rows):
        row_path = f"{path}[{index}]"
        if isinstance(row, DojoRow):
            if counts.get(row.dojo_name, 0) > 0:
                entries.append((row.weight, row.dojo_name))
            continue
        if isinstance(row, DojoGroupRow):
            sub_table = _group_drop_table(row, counts)
        else:
            sub_table = _drop_table_from_rows(
                row.rows, counts, f"{row_path}.table", emptied
            )
        if sub_table is None:
            emptied.append(EmptiedRow(row_path, leaf_dojo_names((row,))))
            continue
        entries.append((row.weight, sub_table))

    # Only zero-weight survivors is as empty as none
    if not any(weight > 0 for weight, _ in entries):
        return None
    return DropTable.of(entries)


def _group_drop_table(
    row: DojoGroupRow, counts: Mapping[str, int]
) -> DropTable[str] | None:
    """One DojoGroupRow's sub-table: each dojo in counts weighted
    count_weight(count, alpha_of(row.within)), or None if every weight is 0.

    Private helper - single caller is _drop_table_from_rows().
    Inputs: row, counts. Output: DropTable[str] | None.
    Side effects: none. Exceptions: none.
    """
    alpha = alpha_of(row.within)
    entries = [
        (count_weight(counts[name], alpha), name)
        for name in row.dojo_names
        if name in counts
    ]
    if not any(weight > 0 for weight, _ in entries):
        return None
    return DropTable.of(entries)
