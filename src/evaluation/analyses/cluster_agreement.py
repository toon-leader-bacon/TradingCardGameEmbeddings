"""ClusterAgreement: cluster the embeddings with no knowledge of the labels,
then measure how well the clusters agree with the labels."""

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.evaluation.analyses.card_sample import CardSample
from src.evaluation.analyses.embedding_analysis import (
    AnalysisResult,
    require_fresh_output_dir,
    write_analysis_result,
)
from src.evaluation.analyses.labeled_sample import draw_labeled_sample
from src.evaluation.embedding.embedding_table import EmbeddingTable
from src.evaluation.labels.card_labels import CardLabels


class ClusterAgreement:
    """KMeans with k = the number of distinct labels in the sample, run on
    unit-length vectors (so distances rank like cosine), then
    permutation-invariant agreement with the labels.

    Scalars: "ari" (adjusted Rand index: 0 = chance, 1 = identical
    partitions), "nmi" (normalized mutual information, 0-1), "n_cards",
    "n_labels". name = "cluster_agreement".

    Inputs (constructor): labels, sample (e.g. PerLabelCap), seed (sampling
        and KMeans initialisation), n_init (KMeans restarts; fixed so runs
        are comparable across tables).
    """

    def __init__(
        self, labels: CardLabels, sample: CardSample, seed: int, n_init: int = 10
    ) -> None:
        """Side effects: none. Exceptions: ValueError if n_init < 1."""
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")
        self.name = "cluster_agreement"
        self._labels = labels
        self._sample = sample
        self._seed = seed
        self._n_init = n_init

    def run(self, table: EmbeddingTable, output_dir: Path) -> AnalysisResult:
        """See EmbeddingAnalysis.run. Writes only scalars.json.

        Example:
            >>> ClusterAgreement(GameLabels(), PerLabelCap(500), seed=0).run(
            ...     table, out / "cluster_agreement").scalars["ari"]
            0.93
        """
        require_fresh_output_dir(output_dir)
        sample = draw_labeled_sample(table, self._labels, self._sample, self._seed)

        # Cluster blind to the labels, into as many clusters as labels
        clusters = self._cluster(sample.unit_vectors(), len(sample.distinct_labels))

        # Score the partition against the labels
        scalars = sample.size_scalars()
        scalars.update(_agreement_scores(list(sample.labels), clusters))
        return write_analysis_result(output_dir, scalars)

    def _cluster(self, vectors: np.ndarray, cluster_count: int) -> np.ndarray:
        """Inputs: unit vectors, cluster_count (>= 2). Output: int cluster id
        per row. Side effects: none. Exceptions: none expected."""
        kmeans = KMeans(
            n_clusters=cluster_count, n_init=self._n_init, random_state=self._seed
        )
        return kmeans.fit_predict(vectors)


def _agreement_scores(labels: list[str], clusters: np.ndarray) -> dict[str, float]:
    """Inputs: the true labels and cluster ids, same length. Output:
    {"ari": ..., "nmi": ...}. Side effects: none. Exceptions: none."""
    return {
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
    }
