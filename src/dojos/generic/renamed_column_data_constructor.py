"""RenamedColumnDataConstructor - a Decorator that lets a generic
DataConstructor read a metric whose uuid column has a different name.

Lives in dojos/generic/ (not under any one game's dojo package) since
it wraps any DataConstructor - it has no metric-family-specific logic
of its own, unlike data_constructors/, where each class maps to one
metric family. fabtcg_decklists and play_gwent's
card_inclusion_dojos.py use it to read their per-faction/per-hero
inclusion columns as PickNumberDecayCurveDataConstructor's
take_rate_by_pick_number / sample_count_by_pick_number. This wrapper
renames the chunk's columns before delegating, so the generic
constructor is reused unchanged.
"""

from typing import Generic, List, Mapping

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructor import DataConstructor
from src.schema.type_hints import LabelT, TrainingDatum


class RenamedColumnDataConstructor(Generic[LabelT]):
    """A DataConstructor that renames chunk columns, then delegates.

    Satisfies the DataConstructor Protocol (../generic/data_constructor.py).
    """

    def __init__(
        self, inner: DataConstructor[LabelT], column_renames: Mapping[str, str]
    ) -> None:
        """
        Inputs:
            inner: the DataConstructor that builds TrainingDatum from the
                renamed chunk.
            column_renames: {metric column name: name inner reads}, e.g.
                {"inclusion_rate_by_hero": "take_rate_by_pick_number"}.
                Copied, never mutated.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if column_renames is empty (the wrapper
            would do nothing) or maps two columns onto one name.

        Example:
            >>> RenamedColumnDataConstructor(
            ...     PickNumberDecayCurveDataConstructor(min_sample_count=5),
            ...     {"inclusion_rate_by_hero": "take_rate_by_pick_number"},
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

    def build(
        self, chunk: pd.DataFrame, lookup: CardLookup
    ) -> List[TrainingDatum[LabelT]]:
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
