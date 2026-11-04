"""Cluster labels: the known category each stored card belongs to under
one grouping (its game, rarity, faction, ...), for the intrinsic analyses
to color by and to score the embedding's clusters against.

A ClusterLabels maps a CardRow to its cluster label, or None when that
card has none under the grouping (the analysis then leaves it out). The
labels are known facts about cards, never clusters an analysis found.
This module holds the Protocol and the two trivial groupings:

- GameClusterLabels: the card's game.
- HoldoutTierClusterLabels: the card's tier under a HoldoutSpec (the
  domain-shift grouping: seen in training, or held out).

MetricParquetClusterLabels (cluster labels read from metric parquets) is
in metric_parquet_cluster_labels.py.
"""

from typing import Protocol

from src.evaluation.intrinsic.card_row import CardRow
from src.schema.holdout import HoldoutSpec


class ClusterLabels(Protocol):
    """The cluster label each card should carry under one grouping.

    grouping: what the cards are grouped by (e.g. "rarity"; its cluster
        labels are then "common", "rare", ...), used in output paths and
        plot legends.
    """

    grouping: str

    def cluster_label_of(self, row: CardRow) -> str | None:
        """Inputs: row (CardRow). Output: the card's cluster label, or None
        if it has none under this grouping. Side effects: none. Exceptions: none."""
        ...


class GameClusterLabels:
    """Every card's cluster label is its game (GameId value). grouping =
    "game"."""

    def __init__(self) -> None:
        """Inputs: none. Output: none (constructor). Side effects: none.
        Exceptions: none."""
        self.grouping = "game"

    def cluster_label_of(self, row: CardRow) -> str:
        """Inputs: row (CardRow). Output: row.source_game.value, never
        None. Side effects: none. Exceptions: none.

        Example:
            >>> GameClusterLabels().cluster_label_of(CardRow(card_id, GameId.GWENT))
            'gwent'
        """
        return row.source_game.value


class HoldoutTierClusterLabels:
    """Every card's cluster label is its CardTier under holdout (typically the
    checkpoint's own, from load_checkpoint_holdout): whether training saw
    it. grouping = "holdout_tier"."""

    def __init__(self, holdout: HoldoutSpec) -> None:
        """Inputs: holdout (HoldoutSpec). Output: none (constructor). Side
        effects: none. Exceptions: none."""
        self.grouping = "holdout_tier"
        self._holdout = holdout

    def cluster_label_of(self, row: CardRow) -> str:
        """Inputs: row (CardRow). Output: the CardTier value ("train",
        "test" or "validation"), never None. Side effects: none.
        Exceptions: none.

        Example:
            >>> HoldoutTierClusterLabels(spec).cluster_label_of(row)
            'validation'
        """
        return self._holdout.tier_of(row.nocab_uuid, row.source_game).value
