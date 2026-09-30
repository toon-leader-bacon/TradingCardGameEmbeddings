"""EffectiveRank: how many dimensions an embedding actually uses."""

from pathlib import Path

import numpy as np

from src.evaluation.analyses.embedding_analysis import (
    AnalysisResult,
    require_fresh_output_dir,
    write_analysis_result,
)
from src.evaluation.embedding.embedding_table import EmbeddingTable

_TOP_DIRECTIONS = 5
# Centered energy below this fraction of the raw energy is rounding, not variance
_RELATIVE_TOLERANCE = 1e-12


class EffectiveRank:
    """Dimensional-collapse check over every card in the table (no labels,
    no sampling: an SVD of a few thousand cards is cheap).

    The vectors are centered. Their squared singular values, normalized to
    sum to 1, give each principal direction's share of the variance, and
    the effective rank is exp(entropy of those shares). That is the number
    of equally used directions with the same spread: 1 for a line, and for
    isotropic noise min(n_cards - 1, embedding width) - so compare
    effective_rank_fraction only between tables with many more cards than
    dimensions. A contrastive head that collapsed
    onto a few directions shows up here long before any labeled metric
    moves (the first Gwent contrastive run fell from 48 to 10).

    Scalars:
        "effective_rank": exp(entropy of the variance shares).
        "effective_rank_fraction": effective_rank / embedding width.
        "top1_variance_share", "top5_variance_share": share of the variance
            in the leading 1 and 5 directions (5 capped at the rank).
        "n_cards", "embedding_dim".
    name = "effective_rank".
    """

    def __init__(self) -> None:
        """Side effects: none. Exceptions: none."""
        self.name = "effective_rank"

    def run(self, table: EmbeddingTable, output_dir: Path) -> AnalysisResult:
        """See EmbeddingAnalysis.run. Writes only scalars.json. Raises
        ValueError (nothing written) with fewer than two cards, or if every
        card has the same vector (no variance to share out).

        Example:
            >>> EffectiveRank().run(table, out / "effective_rank").scalars[
            ...     "effective_rank"]
            10.3
        """
        require_fresh_output_dir(output_dir)
        vectors = table.vectors_for(table.rows()).astype(np.float64)
        if len(vectors) < 2:
            raise ValueError(f"need at least 2 cards, table has {len(vectors)}")

        shares = _variance_shares(vectors)
        scalars = {
            "effective_rank": _effective_rank(shares),
            "top1_variance_share": float(shares[0]),
            "top5_variance_share": float(shares[:_TOP_DIRECTIONS].sum()),
            "n_cards": float(len(vectors)),
            "embedding_dim": float(vectors.shape[1]),
        }
        scalars["effective_rank_fraction"] = (
            scalars["effective_rank"] / vectors.shape[1]
        )
        return write_analysis_result(output_dir, scalars)


def _variance_shares(vectors: np.ndarray) -> np.ndarray:
    """Each principal direction's share of the centered vectors' variance,
    largest first.

    Inputs: vectors (n, d), n >= 2. Output: 1-D array summing to 1.
    Side effects: none.
    Exceptions: ValueError if the vectors have no variance at all (relative
        to their size, so float rounding in the mean does not count).
    """
    centered = vectors - vectors.mean(axis=0)
    energy = np.linalg.svd(centered, compute_uv=False) ** 2
    total = energy.sum()
    if total <= _RELATIVE_TOLERANCE * max(float((vectors**2).sum()), 1.0):
        raise ValueError("every card has the same vector: no variance to measure")
    return energy / total


def _effective_rank(shares: np.ndarray) -> float:
    """exp(Shannon entropy) of shares; zero shares contribute nothing.
    Inputs: shares (sum 1). Output: float in [1, len(shares)].
    Side effects: none. Exceptions: none."""
    positive = shares[shares > 0]
    return float(np.exp(-(positive * np.log(positive)).sum()))
