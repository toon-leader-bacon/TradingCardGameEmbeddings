"""LabelCompactness: how tightly cards group by their known label."""

from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

from src.evaluation.analyses.card_sample import CardSample
from src.evaluation.analyses.embedding_analysis import (
    AnalysisResult,
    require_fresh_output_dir,
    write_analysis_result,
)
from src.evaluation.analyses.labeled_sample import draw_labeled_sample
from src.evaluation.embedding.embedding_table import EmbeddingTable
from src.evaluation.labels.card_labels import CardLabels


class LabelCompactness:
    """Two views of label grouping, on unit-length vectors (Euclidean
    distance between them ranks pairs like cosine distance).

    Needs at least one label with two or more sampled cards (so a
    PerLabelCap(1) sample is refused): both scores are undefined otherwise.

    Scalars:
        "silhouette": mean silhouette of the labels as clusters (-1..1;
            higher = tighter, better-separated labels), Euclidean on unit
            vectors - not numerically a cosine silhouette. A card whose
            label has only one sampled card scores 0.
        "nearest_centroid_accuracy": the fraction of cards whose nearest
            label centroid is their own label's, each card's own label
            centroid computed without it (leave-one-out, so a card never
            votes for itself). Cards of a label with only one sampled card
            have no such centroid and are left out of this score.
        "majority_baseline": the largest label's share of the scored
            cards - what always guessing that label would get.
        "n_scored_cards": how many cards the two scores above cover.
        "n_cards", "n_labels".
    name = "label_compactness".

    Inputs (constructor): labels, sample, seed (sampling only).
    """

    def __init__(self, labels: CardLabels, sample: CardSample, seed: int) -> None:
        """Side effects: none. Exceptions: none."""
        self.name = "label_compactness"
        self._labels = labels
        self._sample = sample
        self._seed = seed

    def run(self, table: EmbeddingTable, output_dir: Path) -> AnalysisResult:
        """See EmbeddingAnalysis.run. Writes only scalars.json. Also raises
        ValueError (nothing written) if no label has two or more sampled
        cards.

        Example:
            >>> LabelCompactness(GameLabels(), PerLabelCap(500), seed=0).run(
            ...     table, out / "label_compactness").scalars["silhouette"]
            0.31
        """
        require_fresh_output_dir(output_dir)
        sample = draw_labeled_sample(table, self._labels, self._sample, self._seed)
        vectors = sample.unit_vectors()
        label_array = np.array(sample.labels)
        # Both scores need some label with a second card to compare against
        _require_a_repeated_label(label_array)

        scalars = sample.size_scalars()
        scalars["silhouette"] = _silhouette(vectors, label_array)
        # Leave-one-out nearest centroid, with its chance baseline
        scalars.update(_nearest_centroid_scalars(vectors, label_array))
        return write_analysis_result(output_dir, scalars)


def _silhouette(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Inputs: unit vectors, labels (>= 2 distinct, some label repeated).
    Output: mean silhouette (Euclidean on unit vectors). Side effects:
    none. Exceptions: none expected given those preconditions."""
    return float(silhouette_score(vectors, labels, metric="euclidean"))


def _nearest_centroid_scalars(
    vectors: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Inputs: unit vectors, labels (some label repeated). Output:
    "nearest_centroid_accuracy", "majority_baseline" and
    "n_scored_cards", over the cards whose label has >= 2 cards (at least
    two, by the precondition). Side effects: none. Exceptions: none.

    Each card is compared with every label's centroid; its own label's
    centroid excludes the card: (sum - card) / (count - 1).
    """
    # Label index, per-label sums and counts
    names, label_index, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    sums = np.zeros((len(names), vectors.shape[1]), dtype=np.float64)
    np.add.at(sums, label_index, vectors)
    centroids = sums / counts[:, None]

    # Squared distance from every card to every label centroid
    distances = _squared_distances(vectors, centroids)

    # Replace each scored card's own-label distance with its leave-one-out one
    scored = counts[label_index] >= 2
    rows = np.flatnonzero(scored)
    own = label_index[rows]
    left_out = (sums[own] - vectors[rows]) / (counts[own] - 1)[:, None]
    distances[rows, own] = np.sum((vectors[rows] - left_out) ** 2, axis=1)

    # Score only the cards that had a leave-one-out centroid
    predicted = np.argmin(distances[rows], axis=1)
    return {
        "nearest_centroid_accuracy": float(np.mean(predicted == own)),
        "majority_baseline": float(counts[counts >= 2].max() / len(rows)),
        "n_scored_cards": float(len(rows)),
    }


def _squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Inputs: points (n, d), centers (k, d). Output: (n, k) squared
    Euclidean distances, float64, clipped at 0 against rounding. Side
    effects: none. Exceptions: none."""
    points64 = points.astype(np.float64)
    result = (
        np.sum(points64**2, axis=1)[:, None]
        - 2.0 * points64 @ centers.T
        + np.sum(centers**2, axis=1)[None, :]
    )
    return np.maximum(result, 0.0)


def _require_a_repeated_label(labels: np.ndarray) -> None:
    """Inputs: labels. Output: None. Side effects: none. Exceptions:
    ValueError if every label occurs exactly once (e.g. a PerLabelCap(1)
    sample)."""
    _, counts = np.unique(labels, return_counts=True)
    if counts.max() < 2:
        raise ValueError(
            "label compactness needs a label with at least 2 sampled cards; "
            f"all {len(counts)} labels have one each"
        )
