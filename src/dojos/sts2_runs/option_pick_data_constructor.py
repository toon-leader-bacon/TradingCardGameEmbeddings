"""OptionPickDataConstructor - rows of any PickChoiceMetric
(src/data_refinement/metrics/sts2_runs/pick_choice_metric.py) -> data for a
MultiGroupOptionSelectionDojo.

The four StS2 pick metrics share one row layout (deck_uuids, offered_uuids,
picked_uuid), so one constructor serves them all.
"""

from typing import List, cast

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructors.row_values import (
    cards_for_uuids,
    option_cards_for_uuids,
    parsed_uuid,
    pick_position,
)
from src.schema.type_hints import MultiCardInput, TrainingDatum


class OptionPickDataConstructor:
    """DataConstructor: one ([option cards, deck cards], label) per row.

    **options MUST stay group index 0, the deck group index 1** (the
    cell's loss baseline counts group 0).

    The label is the picked card's index among the options, or
    len(options) when picked_uuid is NULL: the skip option a dojo built
    with can_skip=True adds. Only the card reward writes NULLs; for the
    other metrics every row has a pick, so the same constructor feeds a
    dojo without can_skip. Pair a NULL-bearing metric only with a dojo
    built with can_skip=True.
    """

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of metric rows into option-selection data.

        Inputs:
            chunk: rows of a PickChoiceMetric's output.
            lookup: the split's holdout-filtered lookup.
        Output: one ([option_cards, deck_cards], label) per usable row.
            A row is skipped when any offered uuid fails to parse or look
            up (it would shift the label), or when picked_uuid is set but
            not among the options. Deck uuids that fail are dropped
            individually; an empty deck is valid.
        Side effects: none.
        Exceptions: none (a bad row is skipped, never raised).

        Example:
            >>> OptionPickDataConstructor().build(chunk, lookup)
            [([[<GenericCard>, <GenericCard>], [<GenericCard>]], 1), ...]
        """
        result: List[TrainingDatum] = []

        # One datum per row that still has all of its offered cards
        for _, row in chunk.iterrows():
            offered = option_cards_for_uuids(lookup, row["offered_uuids"])
            if offered is None:
                continue
            label = _label_of(row["picked_uuid"], offered)
            if label is None:
                continue
            result.append(
                ([offered, cards_for_uuids(lookup, row["deck_uuids"])], label)
            )
        return result


def _label_of(raw_picked_uuid: object, offered: MultiCardInput) -> int | None:
    """The row's label: the picked card's position among the offered
    cards, or len(offered) for a skip.

    Inputs: raw_picked_uuid (the picked_uuid cell: null, as pd.isna sees
        it, means a skip), offered (the row's offered cards).
    Output: int; None if a non-null picked value does not parse or is not
        among the offered cards (the row is then dropped, never
        mislabelled as a skip).
    Side effects: none. Exceptions: none.

    Example:
        >>> _label_of(None, [card_a, card_b])
        2
    """
    # The cell is a string or null; pd.isna covers None, NaN and pd.NA
    if pd.isna(cast("str | None", raw_picked_uuid)):
        return len(offered)
    return pick_position(offered, parsed_uuid(raw_picked_uuid))
