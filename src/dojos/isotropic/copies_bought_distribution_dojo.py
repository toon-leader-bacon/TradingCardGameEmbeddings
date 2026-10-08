"""CopiesBoughtDistributionDojo - card -> the distribution of how many
copies a finished deck holds (CopiesBoughtDistributionMetric).

Each metric row is one observation: this many copies of the card in one
finished deck. Trained with cross entropy, the head learns the whole
distribution P(copies | card), where AverageCopiesBoughtDojo (an MSE head
on the per-card mean) learns only its mean. The card is the only input,
so the best the head can do is the card's empirical distribution.
"""

from pathlib import Path
from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.metrics.isotropic.summary.copies_bought_distribution_metric import (  # noqa: E501
    CopiesBoughtDistributionMetric,
)
from src.dojos.generic.data_constructors.row_values import card_for_uuid
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.single_card_fixed_classification.dojo import (
    SingleCardFixedClassificationDojo,
)
from src.schema.holdout import HoldoutSpec
from src.schema.type_hints import TrainingDatum

# Copies at or above this share one class: 10+ copies are under 1% of rows
# and mostly basic cards (Copper, Estate)
_TOP_COPIES = 10
COPIES_LABEL_VALUES: tuple[str, ...] = (
    *(str(copies) for copies in range(1, _TOP_COPIES)),
    f"{_TOP_COPIES}+",
)


def copies_label(copies: int) -> str | None:
    """The class label for one deck's copy count.

    Inputs: copies (int), the count in one finished deck.
    Output: a COPIES_LABEL_VALUES entry, or None when copies < 1 (a card
        absent from a deck is not an observation).
    Side effects: none. Exceptions: none.

    Example:
        >>> copies_label(3), copies_label(14), copies_label(0)
        ('3', '10+', None)
    """
    if copies < 1:
        return None
    return COPIES_LABEL_VALUES[min(copies, _TOP_COPIES) - 1]


class CopiesBoughtDataConstructor:
    """DataConstructor for (nocab_uuid, copies) rows: card in, copy-count
    class label out. Satisfies the DataConstructor Protocol
    (../generic/data_constructor.py)."""

    def build(self, chunk: pd.DataFrame, lookup: CardLookup) -> List[TrainingDatum]:
        """Convert a chunk of (nocab_uuid, copies) rows into (card, label).

        Inputs: chunk (rows of copies_bought_distribution.parquet), lookup.
        Output: one (GenericCard, str) pair per row whose card looks up
            and whose copy count has a label; other rows are skipped.
        Side effects: none. Exceptions: none.

        Example:
            >>> CopiesBoughtDataConstructor().build(chunk, binder)
            [(<GenericCard Village>, '2'), ...]
        """
        result: List[TrainingDatum] = []

        # Pair each row's card with its bucketed copy count
        for raw_uuid, copies in zip(chunk["nocab_uuid"], chunk["copies"]):
            card = card_for_uuid(lookup, raw_uuid)
            label = copies_label(int(copies))
            if card is None or label is None:
                continue
            result.append((card, label))
        return result


class CopiesBoughtDistributionDojo(SingleCardFixedClassificationDojo):
    """Card -> P(copies in a finished deck | card is in the deck), as a
    class over COPIES_LABEL_VALUES ("1" ... "9", "10+")."""

    METRIC = CopiesBoughtDistributionMetric

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs:
            card_binder: the Dominion binder. holdout, card_embedding_size,
            name, rng_seed, strict_version_check: as CardAverageMetricDojo.
            path_to_training_data: overrides METRIC.DEFAULT_OUTPUT_PATH.
        Output: none (constructor).
        Side effects, exceptions: see SingleCardFixedClassificationDojo.
        """
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=CopiesBoughtDataConstructor(),
            label_values=COPIES_LABEL_VALUES,
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
