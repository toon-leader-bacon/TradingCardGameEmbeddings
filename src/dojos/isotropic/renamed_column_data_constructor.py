"""RenamedColumnDataConstructor - a Decorator that lets a generic
DataConstructor read a metric whose uuid column has a different name.

The isotropic deck-level metrics key their rows by what the row's card
group is (kingdom_uuid, partial_deck_uuid), while
DeckLabelDataConstructor reads a fixed "deck_uuid" column. This wrapper
renames the chunk's columns before delegating, so the generic constructor
is reused unchanged. It is the only place the column names are mapped.

If DeckLabelDataConstructor grows a deck_uuid_column argument (as
CardAverageDataConstructor has uuid_column), this module can be deleted.
"""

from typing import List, Mapping

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructor import DataConstructor
from src.schema.type_hints import TrainingDatum


class RenamedColumnDataConstructor:
    """A DataConstructor that renames chunk columns, then delegates.

    Satisfies the DataConstructor Protocol (../generic/data_constructor.py).
    """

    def __init__(
        self, inner: DataConstructor, column_renames: Mapping[str, str]
    ) -> None:
        """
        Inputs:
            inner: the DataConstructor that builds TrainingDatum from the
                renamed chunk.
            column_renames: {metric column name: name inner reads}, e.g.
                {"kingdom_uuid": "deck_uuid"}. Copied, never mutated.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if column_renames is empty (the wrapper
            would do nothing) or maps two columns onto one name.

        Example:
            >>> RenamedColumnDataConstructor(
            ...     DeckLabelDataConstructor(box, "winner_turns"),
            ...     {"kingdom_uuid": "deck_uuid"},
            ... )
        """
        if not column_renames:
            raise ValueError("column_renames must name at least one column")
        if len(set(column_renames.values())) != len(column_renames):
            raise ValueError(
                f"column_renames maps two columns onto one: {column_renames}"
            )
        self._inner = inner
        self._column_renames = dict(column_renames)

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Rename chunk's columns, then build with the inner constructor.

        Inputs:
            chunk: one chunk of metric rows; columns missing from chunk
                are simply not renamed (the inner constructor then fails
                on them as it would for any malformed chunk).
            lookup: the split's holdout-filtered card lookup.
        Output: List[TrainingDatum], exactly what inner.build returns for
            the renamed chunk.
        Side effects: none - chunk is not modified (rename returns a new
            DataFrame).
        Exceptions: whatever inner.build raises.

        Example:
            >>> constructor.build(chunk, lookup)
            [([<GenericCard>, ...], 19.0), ...]
        """
        result = self._inner.build(chunk.rename(columns=self._column_renames), lookup)
        return result
