"""DropTable: pick one outcome at random, each with its own weight.

A Python port of DropTable<T> from nocabLib (github.com/toon-leader-bacon/
nocabLib, NocabRNG/DropTable.cs), treated as a small external library:
nothing here knows about cards, dojos or training.

Differences from the C# original:
- Immutable. Entries are a tuple fixed at construction, so the total
  weight can never go stale. Build a new table instead of editing one.
- Weights are validated (finite, >= 0, positive total) instead of passed
  through abs(), so a negative weight is reported, not silently flipped.
- The random source is passed to pull(), not stored, so one table can be
  shared by many users, each with its own seeded random.Random.
- An entry's outcome may itself be a DropTable: pulling it pulls again
  from that sub-table ("roll on another table").
- filtered() builds a new table of only the outcomes a predicate accepts,
  renormalizing weights, with emptied sub-tables dropped.
- Float rounding that runs a draw past the last entry returns the last
  entry with a positive weight, not a random one.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar, Union, cast

T = TypeVar("T")


@dataclass(frozen=True)
class DropTableEntry(Generic[T]):
    """One row of a DropTable.

    weight: relative chance of this row (finite, >= 0). A zero-weight row
        is kept but never pulled.
    outcome: what pulling this row yields: a T, or a nested DropTable to
        pull from in turn. An outcome must not be a DropTable meant as a
        plain value; a DropTable outcome always means "roll again".
    """

    weight: float
    outcome: Union[T, "DropTable[T]"]

    def __post_init__(self) -> None:
        # Validate inputs: `not x >= 0` also rejects NaN
        if not (math.isfinite(self.weight) and self.weight >= 0):
            raise ValueError(f"weight must be finite and >= 0, got {self.weight}")


@dataclass(frozen=True)
class DropTable(Generic[T]):
    """A weighted choice over outcomes, possibly nested.

    entries: the rows, in order (order only matters for reproducing a
        draw from the same random value). At least one row must have a
        positive weight.

    Example:
        >>> meta = DropTable.of([(1, "power"), (1, "armor")])
        >>> table = DropTable.of([(8, "nothing"), (2, meta)])
        >>> table.pull(random.Random(1))
        'nothing'
    """

    entries: tuple[DropTableEntry[T], ...]

    def __post_init__(self) -> None:
        # Validate inputs: a table with nothing pullable is an error
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be a tuple (use DropTable.of for pairs)")
        if not any(entry.weight > 0 for entry in self.entries):
            raise ValueError("a DropTable needs at least one positive weight")

    @classmethod
    def of(
        cls, rows: Iterable[tuple[float, Union[T, "DropTable[T]"]]]
    ) -> "DropTable[T]":
        """Build a table from (weight, outcome) pairs.

        Inputs: rows (iterable of (weight, outcome); outcome may be a
            DropTable).
        Output: DropTable.
        Side effects: none (rows is consumed once).
        Exceptions: ValueError as DropTable and DropTableEntry validate.

        Example:
            >>> DropTable.of([(3, "a"), (1, "b")])
        """
        return cls(tuple(DropTableEntry(weight, outcome) for weight, outcome in rows))

    @classmethod
    def uniform(cls, outcomes: Iterable[T]) -> "DropTable[T]":
        """A table giving every outcome the same weight.

        Inputs: outcomes (iterable, at least one).
        Output: DropTable.
        Side effects: none.
        Exceptions: ValueError if outcomes is empty.

        Example:
            >>> DropTable.uniform(["a", "b", "c"]).pull(random.Random(1))
        """
        rows = [(1.0, outcome) for outcome in outcomes]
        if not rows:
            raise ValueError("a uniform DropTable needs at least one outcome")
        return cls.of(rows)

    @property
    def total_weight(self) -> float:
        """Sum of this table's own entry weights (sub-tables count as
        their entry's weight, not their contents). Side effects: none."""
        return sum(entry.weight for entry in self.entries)

    def pull(self, rng: random.Random) -> T:
        """Draw one outcome, rolling into sub-tables until a plain outcome.

        Each level draws a point in [0, total_weight) and walks its
        entries subtracting weights until the point falls inside one. A
        loop over levels, not recursion: nesting depth is data-driven.

        Inputs: rng (random.Random; its state advances once per level).
        Output: T.
        Side effects: advances rng.
        Exceptions: none (construction guarantees a pullable entry at
            every level).

        Example:
            >>> DropTable.of([(1, "a"), (0, "b")]).pull(random.Random(0))
            'a'
        """
        table: DropTable[T] = self
        # Roll on each table until a plain outcome comes up
        while True:
            outcome = table._entry_at(rng.random() * table.total_weight).outcome
            if not isinstance(outcome, DropTable):
                return outcome
            table = outcome

    def filtered(self, keep: Callable[[T], bool]) -> "DropTable[T] | None":
        """A new table holding only the outcomes keep() accepts.

        Rows whose plain outcome keep() rejects are dropped, sub-tables
        are filtered the same way, and a sub-table left with nothing
        pullable is dropped with its row. Surviving weights are kept as
        is, so their relative odds are unchanged (renormalized over what
        survives). Recursive over sub-tables: the nesting mirrors the
        table's own shape, which reads more plainly than an explicit stack.

        Inputs: keep (predicate over plain outcomes).
        Output: the filtered DropTable, or None if nothing pullable
            survives.
        Side effects: none (self is unchanged); calls keep once per plain
            outcome.
        Exceptions: whatever keep raises.

        Example:
            >>> DropTable.uniform([1, 2, 3, 4]).filtered(lambda n: n % 2 == 0)
            DropTable(entries=(DropTableEntry(weight=1.0, outcome=2), ...))
        """
        result_rows: list[DropTableEntry[T]] = []

        # Keep each row whose outcome (or filtered sub-table) survives
        for entry in self.entries:
            survivor = _surviving_entry(entry, keep)
            if survivor is not None:
                result_rows.append(survivor)

        # Nothing pullable left means no table at all
        if not any(row.weight > 0 for row in result_rows):
            return None
        return DropTable(tuple(result_rows))

    def _entry_at(self, point: float) -> DropTableEntry[T]:
        """The entry whose weight span contains point in [0, total_weight).

        Float rounding can leave point at or past the end of the last
        span; the last positive-weight entry is returned then.
        Inputs: point. Output: DropTableEntry. Side effects: none.
        Exceptions: none.
        """
        last_positive: DropTableEntry[T] | None = None
        # Walk the spans until point falls inside one
        for entry in self.entries:
            if entry.weight <= 0:
                continue
            last_positive = entry
            if point < entry.weight:
                return entry
            point -= entry.weight
        # Construction guarantees a positive entry, so this is never None
        return cast(DropTableEntry[T], last_positive)


def _surviving_entry(
    entry: DropTableEntry[T], keep: Callable[[T], bool]
) -> DropTableEntry[T] | None:
    """entry if it survives filtering, else None. A plain outcome survives
    when keep() accepts it (entry returned as is); a sub-table when its
    filtered copy is not None (a new entry, same weight, holding the copy).
    Returns an entry, not an outcome, so a plain None outcome can never be
    mistaken for "rejected".

    Inputs: entry, keep. Output: DropTableEntry or None.
    Side effects: calls keep. Exceptions: whatever keep raises.
    """
    if isinstance(entry.outcome, DropTable):
        sub_table = entry.outcome.filtered(keep)
        return None if sub_table is None else DropTableEntry(entry.weight, sub_table)
    return entry if keep(cast(T, entry.outcome)) else None


# Keep the module's public surface explicit for the "external library" use
__all__ = ["DropTable", "DropTableEntry"]
