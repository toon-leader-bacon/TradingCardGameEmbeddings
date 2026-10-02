"""GroupLabelDataConstructor - two CardGroups and one label column ->
([group_0, group_1], label), for the multi-group binary and regression
cells.

Serves every isotropic row shape with two card groups and one value:
deck pairs (deck_uuid_lo, deck_uuid_hi), [partial deck, kingdom],
[opening buy, kingdom], and the per-(kingdom, card) rows as
[[card], kingdom].
"""

from typing import Callable, List

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructors.row_values import cast_to_float
from src.dojos.isotropic.card_groups import CardGroup
from src.schema.type_hints import MultiGroupInput, TrainingDatum


class GroupLabelDataConstructor:
    """DataConstructor: ([group_0.cards, group_1.cards], label_caster(label)).

    A row is skipped when either group comes out empty (a hidden card, a
    missing deck): GroupSwapMod needs both sides, and an empty group 0
    breaks input_shape_of(). A NaN label is skipped too.
    """

    def __init__(
        self,
        group_0: CardGroup,
        group_1: CardGroup,
        label_column: str,
        label_caster: Callable[[object], float] = cast_to_float,
    ) -> None:
        """
        Inputs:
            group_0, group_1: how to read each group from a row (order
                is the input order the cell sees).
            label_column: the row's label column.
            label_caster: raw label cell -> float; the default casts
                bool/int/float (BceLoss and MseLoss both want floats).
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        self._groups = (group_0, group_1)
        self._label_column = label_column
        self._label_caster = label_caster

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of metric rows into ([group_0, group_1], float)
        data, skipping rows as the class docstring says.

        Inputs: chunk (metric rows), lookup (the split's holdout-filtered
            lookup).
        Output: List[TrainingDatum].
        Side effects: reads the groups' DeckBox. Exceptions: whatever
            label_caster raises on a malformed label.

        Example:
            >>> GroupLabelDataConstructor(
            ...     DeckColumnGroup(box, "deck_uuid_lo"),
            ...     DeckColumnGroup(box, "deck_uuid_hi"),
            ...     "lo_won",
            ... ).build(chunk, lookup)
            [([[<GenericCard>, ...], [<GenericCard>, ...]], 1.0), ...]
        """
        result: List[TrainingDatum] = []
        for _, row in chunk.iterrows():
            raw_label = row[self._label_column]
            if pd.isna(raw_label):
                continue
            groups: MultiGroupInput = [
                group.cards(row, lookup) for group in self._groups
            ]
            if not all(groups):
                continue
            result.append((groups, self._label_caster(raw_label)))
        return result
