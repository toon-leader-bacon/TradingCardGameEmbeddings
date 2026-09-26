"""DataConstructor for PackToPickChoiceSetMetric - see
src/data_refinement/metrics/seventeenlands/draft_data/pack_to_pick_choice_set_metric.py.
"""

from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructors._row_values import (
    _option_cards_and_pick_index,
)
from src.schema.type_hints import TrainingDatum


class PackToPickChoiceSetDataConstructor:
    """DataConstructor for PackToPickChoiceSetMetric (see
    src/data_refinement/metrics/seventeenlands/draft_data/
    pack_to_pick_choice_set_metric.py) - a ragged pack of option cards
    in, the index of the picked option out. Feeds
    MultiCardOptionSelectionDojo (src/dojos/generic/
    multi_card_option_selection/dojo.py): TrainingInput = MultiCardInput
    (just the pack options, no conditioning group - see
    PoolConditionedPickDataConstructor below for the two-group sibling
    built from the near-identical metric of the same name).

    This is the one metric in its own family (no siblings), so there's
    no label_column to parameterize, unlike CardAverageDataConstructor.
    """

    def __init__(self) -> None:
        """
        Inputs:
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of (pack_option_uuids, pick_uuid) rows into
        (MultiCardInput, int) TrainingDatum pairs.

        Inputs:
            chunk: one chunk of rows from PackToPickChoiceSetMetric's
                output parquet file.
        Output: one (option cards, picked option's index) TrainingDatum
            per row - see _option_cards_and_pick_index()'s docstring for
            the exact skip conditions (null/unmatched pick, or any
            option failing to resolve).
        Side effects: none.
        Exceptions: none expected (per-row failures are skipped, not
            raised - mirrors every other DataConstructor in this
            package).

        Example:
            >>> constructor = PackToPickChoiceSetDataConstructor()
            >>> constructor.build(chunk, lookup)
            [([<GenericCard>, <GenericCard>, <GenericCard>], 1), ...]
        """
        results: List[TrainingDatum] = []

        # Resolve each row's option list and picked index together (one
        # row's label is meaningless without the other) via the shared
        # helper both option-selection DataConstructors use; skip rows
        # that fail.
        for _, row in chunk.iterrows():
            option_pick = _option_cards_and_pick_index(
                lookup, row["pack_option_uuids"], row["pick_uuid"]
            )
            if option_pick is None:
                continue
            option_cards, pick_index = option_pick
            results.append((option_cards, pick_index))

        return results
