"""LabeledSample: the sampled, labeled cards an analysis measures, with
their vectors - the preparation every label-using analysis shares."""

from dataclasses import dataclass

import numpy as np

from src.evaluation.analyses.card_sample import CardSample
from src.evaluation.card_row import CardRow
from src.evaluation.embedding.embedding_table import EmbeddingTable
from src.evaluation.labels.card_labels import CardLabels


@dataclass(frozen=True)
class LabeledSample:
    """rows[i] has vectors[i] and labels[i].

    vectors: float32 (len(rows), embedding_dim).
    labels: one label per row; never None (unlabeled cards are dropped).
    """

    rows: tuple[CardRow, ...]
    vectors: np.ndarray
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        """Enforce the row alignment every analysis relies on.
        Exceptions: ValueError unless vectors is 2-D and rows, vectors and
        labels have the same length."""
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        lengths = (len(self.rows), self.vectors.shape[0], len(self.labels))
        if len(set(lengths)) != 1:
            raise ValueError(f"rows, vectors and labels differ in length: {lengths}")

    @property
    def distinct_labels(self) -> tuple[str, ...]:
        """Inputs: none. Output: the distinct labels, sorted. Side effects:
        none. Exceptions: none."""
        return tuple(sorted(set(self.labels)))

    def unit_vectors(self) -> np.ndarray:
        """A method, not a property: it allocates a new array each call.

        Inputs: none. Output: vectors scaled to unit length (a zero vector
        stays zero), so Euclidean distances rank pairs as cosine distance
        does (the values differ: sqrt(2 - 2cos)).
        Side effects: none. Exceptions: none."""
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        result = np.zeros_like(self.vectors)
        np.divide(self.vectors, norms, out=result, where=norms > 0)
        return result

    def size_scalars(self) -> dict[str, float]:
        """The sample-size numbers every analysis reports.

        Inputs: none. Output: {"n_cards": ..., "n_labels": ...}.
        Side effects: none. Exceptions: none."""
        return {
            "n_cards": float(len(self.rows)),
            "n_labels": float(len(self.distinct_labels)),
        }


def draw_labeled_sample(
    table: EmbeddingTable, labels: CardLabels, sample: CardSample, seed: int
) -> LabeledSample:
    """Sample a table's cards, drop unlabeled ones, and read their vectors.

    Inputs: table (open), labels, sample (the rule), seed.
    Output: LabeledSample of the chosen, labeled cards, in table order.
    Side effects: reads table.
    Exceptions: ValueError if fewer than two cards or fewer than two
        distinct labels remain (the message says which, and how many);
        whatever sample.select raises.

    Example:
        >>> sample = draw_labeled_sample(table, GameLabels(), PerLabelCap(500), 0)
        >>> sample.distinct_labels
        ('gwent', 'mtg')
    """
    # Choose rows, then keep only labeled ones
    chosen = sample.select(table.rows(), labels, seed)
    labeled = [(row, labels.label_of(row)) for row in chosen]
    kept = [(row, label) for row, label in labeled if label is not None]

    # Refuse a sample no label-using analysis can measure
    _require_measurable(len(kept), {label for _, label in kept})

    rows = tuple(row for row, _ in kept)
    return LabeledSample(
        rows=rows,
        vectors=table.vectors_for(rows),
        labels=tuple(label for _, label in kept),
    )


def _require_measurable(card_count: int, distinct: set[str]) -> None:
    """Inputs: the kept card count and distinct labels. Output: None.
    Side effects: none. Exceptions: ValueError naming the count if fewer
    than two cards or two distinct labels."""
    if card_count < 2:
        raise ValueError(f"need at least 2 labeled cards, sampled {card_count}")
    if len(distinct) < 2:
        raise ValueError(f"need at least 2 distinct labels, sampled {sorted(distinct)}")
