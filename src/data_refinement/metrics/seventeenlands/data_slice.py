"""SeventeenLandsSlice - which sets and formats of the 17lands corpus a
dojo trains on.

A slice is a filter, never a grouping: a count-table slice always
collapses to one row per key (one label per card or deck). Training the
same metric per format means several dojos with several slices, not one
dojo with several labels per card.

Set-level holdout (a later split policy) is not part of a slice; a
row-stream slice file keeps set and format columns so that split can
group on them.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data_refinement.metrics.seventeenlands.partition import (
    SeventeenLandsPartition,
)
from src.data_retrieval.seventeenlands.refs import Expansion, FormatCode

ALL_SLICE_NAME = "all"


@dataclass(frozen=True)
class SeventeenLandsSlice:
    """A set filter and a format filter.

    expansions: the sets to include; None includes every set.
    formats: the formats to include; None includes every format.
    An empty frozenset is rejected (it would select nothing).
    """

    expansions: frozenset[Expansion] | None = None
    formats: frozenset[FormatCode] | None = None

    def __post_init__(self) -> None:
        """Reject an empty filter.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError if expansions or formats is an empty
            frozenset (None is the way to say "every").
        """
        if self.expansions is not None and not self.expansions:
            raise ValueError("expansions is empty; use None for every set")
        if self.formats is not None and not self.formats:
            raise ValueError("formats is empty; use None for every format")

    @property
    def name(self) -> str:
        """A stable, filename-safe name: "all" for no filter, else
        "sets-<A+B>" and/or "formats-<X+Y>" joined by "_", each list
        sorted.

        Inputs: none. Output: str.
        Side effects: none. Exceptions: none.

        Example:
            >>> SeventeenLandsSlice(formats=frozenset({FormatCode.PremierDraft})).name
            'formats-PremierDraft'
        """
        parts: list[str] = []
        if self.expansions is not None:
            parts.append("sets-" + "+".join(sorted(e.value for e in self.expansions)))
        if self.formats is not None:
            parts.append("formats-" + "+".join(sorted(f.value for f in self.formats)))
        return "_".join(parts) or ALL_SLICE_NAME

    def includes(self, partition: SeventeenLandsPartition) -> bool:
        """Whether partition's set and format pass both filters.

        Inputs: partition. Output: bool.
        Side effects: none. Exceptions: none.

        Example:
            >>> SeventeenLandsSlice().includes(partition)
            True
        """
        set_ok = self.expansions is None or partition.expansion in self.expansions
        format_ok = self.formats is None or partition.format in self.formats
        return set_ok and format_ok
