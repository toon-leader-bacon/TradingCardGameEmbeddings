"""CardSample: which stored cards an analysis looks at.

Corpora reach 10^4-10^5+ cards, and silhouette, clustering and plotting
are super-linear or visually saturated at that size, so every analysis
takes an explicit, seeded sampling rule. The same rows + labels + seed
always select the same cards, so two tables over the same corpus are
sampled identically (EmbeddingTable.rows() is sorted by nocab_uuid).
"""

import random
from typing import Protocol, Sequence

from src.evaluation.card_row import CardRow
from src.evaluation.labels.card_labels import CardLabels


class CardSample(Protocol):
    """A seeded rule choosing a subset of rows."""

    def select(
        self, rows: Sequence[CardRow], labels: CardLabels | None, seed: int
    ) -> list[CardRow]:
        """Inputs: rows (sorted, as EmbeddingTable.rows() returns them),
        labels (None for a label-free analysis), seed.
        Output: the chosen rows, in rows' order.
        Side effects: none. Exceptions: rule-specific ValueError."""
        ...


class AllCards:
    """Every row (labels and seed are ignored; unlabeled cards are dropped
    later by the analysis, not here)."""

    def select(
        self, rows: Sequence[CardRow], labels: CardLabels | None, seed: int
    ) -> list[CardRow]:
        """Inputs: rows, labels (ignored), seed (ignored). Output: list(rows).
        Side effects: none. Exceptions: none.

        Example:
            >>> AllCards().select(rows, None, seed=0) == list(rows)
            True
        """
        return list(rows)


class PerLabelCap:
    """At most max_per_label cards of each label, chosen at random (seeded);
    unlabeled cards are never chosen. Keeps one dominant label (e.g. MTG in
    a game-labeled corpus) from swamping the rest.

    Exceptions: ValueError on construction if max_per_label < 1.
    """

    def __init__(self, max_per_label: int) -> None:
        """Inputs: max_per_label (>= 1). Output: none (constructor).
        Side effects: none. Exceptions: ValueError if max_per_label < 1."""
        if max_per_label < 1:
            raise ValueError(f"max_per_label must be >= 1, got {max_per_label}")
        self.max_per_label = max_per_label

    def select(
        self, rows: Sequence[CardRow], labels: CardLabels | None, seed: int
    ) -> list[CardRow]:
        """Inputs: rows, labels (required), seed.
        Output: the chosen rows, in rows' order.
        Side effects: none.
        Exceptions: ValueError if labels is None.

        Example:
            >>> len(PerLabelCap(2).select(rows, GameLabels(), seed=0))  # 2 games
            4
        """
        if labels is None:
            raise ValueError("PerLabelCap needs labels")
        rng = random.Random(seed)
        chosen: set[CardRow] = set()
        # Group rows by label; label order is sorted so the rng is consumed
        # in the same order every time
        by_label = _rows_by_label(rows, labels)
        for label in sorted(by_label):
            group = by_label[label]
            chosen.update(rng.sample(group, min(len(group), self.max_per_label)))
        # Keep the input order, independent of which label drew first
        return [row for row in rows if row in chosen]


def _rows_by_label(
    rows: Sequence[CardRow], labels: CardLabels
) -> dict[str, list[CardRow]]:
    """Inputs: rows, labels. Output: label -> its rows, in rows' order;
    unlabeled rows omitted. Side effects: none. Exceptions: none."""
    result: dict[str, list[CardRow]] = {}
    for row in rows:
        label = labels.label_of(row)
        if label is not None:
            result.setdefault(label, []).append(row)
    return result
