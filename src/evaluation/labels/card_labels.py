"""Card labels: a categorical label per stored card, for the intrinsic
analyses to color, cluster and score by.

A CardLabels maps a CardRow to a label string, or None when that card is
not labeled (the analysis then leaves it out). This module holds the
Protocol and the two trivial sources:

- GameLabels: the card's game.
- HoldoutTierLabels: the card's tier under a HoldoutSpec (the
  domain-shift label: seen in training, or held out).

MetricParquetLabels (labels read from metric parquets) is in
metric_parquet_labels.py.
"""

from typing import Protocol

from src.evaluation.card_row import CardRow
from src.schema.holdout import HoldoutSpec


class CardLabels(Protocol):
    """A categorical label per card.

    name: what the labels are, used in output paths and plot legends.
    """

    name: str

    def label_of(self, row: CardRow) -> str | None:
        """Inputs: row (CardRow). Output: the card's label, or None if it
        is not labeled. Side effects: none. Exceptions: none."""
        ...


class GameLabels:
    """Every card labeled by its game (GameId value). name = "game"."""

    def __init__(self) -> None:
        """Inputs: none. Output: none (constructor). Side effects: none.
        Exceptions: none."""
        self.name = "game"

    def label_of(self, row: CardRow) -> str:
        """Inputs: row (CardRow). Output: row.source_game.value, never
        None. Side effects: none. Exceptions: none.

        Example:
            >>> GameLabels().label_of(CardRow(card_id, GameId.GWENT))
            'gwent'
        """
        return row.source_game.value


class HoldoutTierLabels:
    """Every card labeled by its CardTier under holdout (typically the
    checkpoint's own, from load_checkpoint_holdout): whether training saw
    it. name = "holdout_tier"."""

    def __init__(self, holdout: HoldoutSpec) -> None:
        """Inputs: holdout (HoldoutSpec). Output: none (constructor). Side
        effects: none. Exceptions: none."""
        self.name = "holdout_tier"
        self._holdout = holdout

    def label_of(self, row: CardRow) -> str:
        """Inputs: row (CardRow). Output: the CardTier value ("train",
        "test" or "validation"), never None. Side effects: none.
        Exceptions: none.

        Example:
            >>> HoldoutTierLabels(spec).label_of(row)
            'validation'
        """
        return self._holdout.tier_of(row.nocab_uuid, row.source_game).value
