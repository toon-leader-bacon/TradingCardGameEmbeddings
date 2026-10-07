"""DataConstructor for the cross-game rarity tier metric - see
src/data_refinement/metrics/cross_game/rarity/rarity_tier_metric.py."""

from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructors.masked_field import MaskedFieldDataConstructor
from src.schema.rarity_tier import RarityTier
from src.schema.type_hints import TrainingDatum


def is_trainable_row(chunk: pd.DataFrame) -> pd.Series:
    """Which rows of a metric chunk the dojo may train on: every row whose
    label is not OTHER. The one definition both the data constructor and
    the TRAIN game balancing use.

    Inputs: chunk (rows with a label column). Output: boolean mask, one
    per row. Side effects: none. Exceptions: KeyError without a label column.
    """
    return chunk["label"] != RarityTier.OTHER.value


class RarityTierDataConstructor:
    """(nocab_uuid, label) rows -> (card, tier string) data, minus the
    OTHER rows.

    A Decorator over MaskedFieldDataConstructor, whose row shape this
    metric shares: OTHER rows (curses, statuses, tokens, quests) are
    dropped before building, because that label is mostly "this is an StS2
    non-card" - a shortcut to the card's game, not a scarcity. The metric
    keeps them for evaluation.

    Inputs (constructor): none. Output: n/a. Side effects: none.
    Exceptions: none.
    """

    def __init__(self) -> None:
        self._inner = MaskedFieldDataConstructor("label")

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of metric rows into TrainingDatum pairs.

        Inputs: chunk (rows with nocab_uuid and label columns), lookup
            (the split's holdout-filtered card lookup, spanning games).
        Output: one (GenericCard, tier string) per non-OTHER row whose card
            the lookup holds; rows for hidden or unknown cards are skipped.
        Side effects: none.
        Exceptions: none expected.

        Example:
            >>> RarityTierDataConstructor().build(chunk, lookup)[0][1]
            'tier_3'
        """
        return self._inner.build(chunk[is_trainable_row(chunk)], lookup)
