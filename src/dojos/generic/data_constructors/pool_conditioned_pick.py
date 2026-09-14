"""DataConstructor for PoolConditionedPickMetric - see
src/data_refinement/metrics/seventeenlands/draft_data/pool_conditioned_pick_metric.py.
"""

from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors._uuid_resolution import (
    _cards_for_uuids,
    _option_cards_and_pick_index,
)
from src.schema.type_hints import MultiGroupInput, TrainingDatum


class PoolConditionedPickDataConstructor:
    """DataConstructor for PoolConditionedPickMetric (see
    src/data_refinement/metrics/seventeenlands/draft_data/
    pool_conditioned_pick_metric.py) - a ragged pack of option cards
    PLUS the drafter's pool-so-far in, the index of the picked option
    (within the pack options only) out. Feeds MultiGroupOptionSelectionDojo
    (src/dojos/generic/multi_group_option_selection/dojo.py):
    TrainingInput = MultiGroupInput = [pack_option_cards, pool_cards].

    **pack options MUST stay group index 0, pool MUST stay index 1.**
    src/schema/type_hints.py's input_shape_of() classifies a
    TrainingInput's shape by peeking group 0 only, and raises ValueError
    the instant it finds an empty list there - it never visits index 1
    at all. A pack's option list is never empty (a pick always comes
    from a non-empty pack); a drafter's pool-so-far IS legitimately
    empty on a draft's very first pick. Putting pool at index 1 means
    that real, common case never trips input_shape_of; reordering these
    groups reintroduces a ValueError on every first-pick row - see that
    function's own docstring before "fixing" this ordering.
    """

    def __init__(self, card_binder: CardBinder) -> None:
        """
        Inputs:
            card_binder: registry to resolve each row's pool_uuids,
                pack_option_uuids, and pick_uuid against. Never written
                to.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._card_binder = card_binder

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """Convert a chunk of (pool_uuids, pack_option_uuids, pick_uuid)
        rows into (MultiGroupInput, int) TrainingDatum pairs.

        Inputs:
            chunk: one chunk of rows from PoolConditionedPickMetric's
                output parquet file.
        Output: one ([option_cards, pool_cards], picked option's index)
            TrainingDatum per row. option_cards/pick_index skip
            conditions are identical to
            PackToPickChoiceSetDataConstructor's (see
            _option_cards_and_pick_index()'s docstring). pool_cards may
            be empty (a valid, expected row - the first pick of a
            draft) via _cards_for_uuids(), which drops individual
            unmatched pool uuids rather than skipping the row - an
            empty pool is never itself a skip condition here.
        Side effects: none.
        Exceptions: none expected (per-row failures are skipped, not
            raised - mirrors every other DataConstructor in this
            package).

        Example:
            >>> constructor = PoolConditionedPickDataConstructor(card_binder)
            >>> constructor.build(chunk)
            [([[<GenericCard>, <GenericCard>], [<GenericCard>]], 0), ...]
        """
        results: List[TrainingDatum] = []

        # Handle the option side exactly like
        # PackToPickChoiceSetDataConstructor.build(); skip rows that
        # fail. The pool side is looked up independently and tolerates
        # being empty - see class/build() docstrings.
        for _, row in chunk.iterrows():
            option_pick = _option_cards_and_pick_index(
                self._card_binder, row["pack_option_uuids"], row["pick_uuid"]
            )
            if option_pick is None:
                continue
            option_cards, pick_index = option_pick
            pool_cards = _cards_for_uuids(self._card_binder, row["pool_uuids"])
            group: MultiGroupInput = [option_cards, pool_cards]
            results.append((group, pick_index))

        return results
