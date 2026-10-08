"""CardRewardPickDataConstructor - CardRewardPickMetric rows -> data for
a MultiGroupOptionSelectionDojo with can_skip=True.

See src/data_refinement/metrics/sts2_runs/card_reward_pick_metric.py for
the row: deck_uuids (the deck on arrival), offered_uuids (the cards
offered) and picked_uuid (NULL = the player took none of them).
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


class CardRewardPickDataConstructor:
    """DataConstructor: one ([offered cards, deck cards], label) per row.

    **offered MUST stay group index 0, the deck group index 1** (the
    cell's loss baseline counts group 0; see
    PoolConditionedPickDataConstructor for why a possibly-empty group
    goes second).

    The label is the picked card's index among the offered cards, or
    len(offered) for a skip, the extra option MultiGroupOptionSelectionDojo
    adds when can_skip is set. Pair it only with a dojo built with
    can_skip=True (a test builds both and checks the label fits the
    logits).
    """

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of metric rows into option-selection data.

        Inputs:
            chunk: rows of CardRewardPickMetric's output.
            lookup: the split's holdout-filtered lookup.
        Output: one ([offered_cards, deck_cards], label) per usable row.
            A row is skipped when any offered uuid fails to parse or
            look up (it would shift the label), or when picked_uuid is
            set but not among the offered cards. Deck uuids that fail
            are dropped individually; an empty deck is valid.
        Side effects: none.
        Exceptions: none (a bad row is skipped, never raised).

        Example:
            >>> CardRewardPickDataConstructor().build(chunk, lookup)
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
