"""PpmiSvdMethod: positive pointwise mutual information over the
co-occurrence counts, then a truncated SVD (Levy & Goldberg's view of
word2vec as implicit matrix factorisation).

PMI(i, j) = log(P(i, j) / (P(i) P_alpha(j))), where P_alpha(j) is the
context distribution raised to alpha (context smoothing) and
renormalised. Negatives are clipped to 0 (PPMI), only the nonzero pairs
are ever computed (the matrix stays sparse), and the vectors are the top
dim left singular vectors scaled by sqrt(singular value).
"""

import numpy as np
import scipy.sparse
from sklearn.utils.extmath import randomized_svd

from src.teacher.cooccurrence import CooccurrenceCounts
from src.teacher.teacher_settings import PpmiSvdOptions

# Seeds the randomized SVD, so a teacher is a function of its inputs
_SVD_SEED = 0


class PpmiSvdMethod:
    """TeacherMethod: PPMI + truncated SVD, with fixed options."""

    def __init__(self, options: PpmiSvdOptions) -> None:
        """Inputs: options. Output: none. Side effects: none.
        Exceptions: none."""
        self._options = options

    def fit(self, counts: CooccurrenceCounts) -> np.ndarray:
        """See TeacherMethod.fit. dim is capped at card count - 1 (a
        truncated SVD needs k < n); a card with no co-occurrence gets a
        zero row.

        Inputs: counts. Output: float32 (cards, min(dim, cards - 1)).
        Side effects: none.
        Exceptions: ValueError if counts hold fewer than two cards or no
            pair.

        Example:
            >>> PpmiSvdMethod(PpmiSvdOptions(dim=64)).fit(counts).shape
            (30123, 64)
        """
        # Validate inputs: an SVD needs two cards and one pair
        _require_fittable(counts)
        dim = min(self._options.dim, len(counts.card_uuids) - 1)

        # Sparse PPMI, then its top singular vectors
        ppmi = _ppmi_of(counts.pair_counts, self._options.context_smoothing)
        left, singular_values = _truncated_svd(ppmi, dim)
        result = (left * np.sqrt(singular_values)).astype(np.float32)
        return result


def _require_fittable(counts: CooccurrenceCounts) -> None:
    """Raise ValueError if counts hold fewer than two cards or no pair.

    Private helper - single caller is PpmiSvdMethod.fit.
    Inputs: counts. Output: None. Side effects: none.
    Exceptions: ValueError naming the game.
    """
    if len(counts.card_uuids) < 2:
        raise ValueError(
            f"{counts.source_game.value}: a teacher needs at least 2 cards, "
            f"got {len(counts.card_uuids)}"
        )
    if counts.pair_counts.nnz == 0:
        raise ValueError(f"{counts.source_game.value}: no two cards share a deck")


def _ppmi_of(
    pair_counts: scipy.sparse.csr_matrix, context_smoothing: float
) -> scipy.sparse.csr_matrix:
    """The PPMI matrix over pair_counts' nonzero entries (log of the
    ratio, clipped at 0, explicit zeros removed).

    Private helper - single caller is PpmiSvdMethod.fit.
    Inputs: pair_counts (symmetric CSR), context_smoothing (alpha).
    Output: float64 CSR, same sparsity or sparser. Side effects: none.
    Exceptions: none.
    """
    pairs = pair_counts.tocoo()
    total = float(pairs.data.sum())

    # Marginals: P(i) from row sums; the context distribution smoothed by alpha
    row_sums = np.asarray(pair_counts.sum(axis=1)).ravel()
    smoothed = row_sums**context_smoothing
    context_probabilities = smoothed / smoothed.sum()

    # PMI on the nonzero pairs only, then clip negatives away
    pmi = (
        np.log(pairs.data / total)
        - np.log(row_sums[pairs.row] / total)
        - np.log(context_probabilities[pairs.col])
    )
    result = scipy.sparse.csr_matrix(
        (np.maximum(pmi, 0.0), (pairs.row, pairs.col)), shape=pair_counts.shape
    )
    result.eliminate_zeros()
    return result


def _truncated_svd(
    matrix: scipy.sparse.csr_matrix, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """matrix's top dim left singular vectors and singular values,
    largest first, from a seeded randomized SVD (deterministic).

    Private helper - single caller is PpmiSvdMethod.fit.
    Inputs: matrix (n x n), dim (< n).
    Output: (float64 (n, dim), float64 (dim,)). Side effects: none.
    Exceptions: none.
    """
    left, singular_values, _ = randomized_svd(
        matrix, n_components=dim, random_state=_SVD_SEED
    )
    return left, singular_values
