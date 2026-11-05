from uuid import uuid4

import numpy as np
import pytest
import scipy.sparse

from src.schema.game_id import GameId
from src.teacher.cooccurrence import CooccurrenceCounts
from src.teacher.ppmi_svd import PpmiSvdMethod, _ppmi_of
from src.teacher.teacher_method import teacher_method_for
from src.teacher.teacher_settings import PpmiSvdOptions


def _counts(pairs: np.ndarray) -> CooccurrenceCounts:
    matrix = scipy.sparse.csr_matrix(pairs.astype(np.float64))
    return CooccurrenceCounts(
        source_game=GameId.MTG,
        card_uuids=tuple(uuid4() for _ in range(pairs.shape[0])),
        pair_counts=matrix,
        deck_counts=np.ones(pairs.shape[0], dtype=np.int64),
        deck_count=1,
    )


def _two_clusters() -> np.ndarray:
    """Cards 0-2 always together, cards 3-5 always together, one weak link."""
    pairs = np.zeros((6, 6))
    for group in ((0, 1, 2), (3, 4, 5)):
        for first in group:
            for second in group:
                if first != second:
                    pairs[first, second] = 10
    pairs[2, 3] = pairs[3, 2] = 1
    return pairs


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))


class TestPpmiSvd:
    def test_cards_in_a_cluster_are_closer_than_across(self) -> None:
        vectors = PpmiSvdMethod(PpmiSvdOptions(dim=3)).fit(_counts(_two_clusters()))

        assert _cosine(vectors[0], vectors[1]) > _cosine(vectors[0], vectors[4])
        assert _cosine(vectors[4], vectors[5]) > _cosine(vectors[1], vectors[5])

    def test_shape_and_dtype(self) -> None:
        vectors = PpmiSvdMethod(PpmiSvdOptions(dim=3)).fit(_counts(_two_clusters()))

        assert vectors.shape == (6, 3)
        assert vectors.dtype == np.float32

    def test_dim_is_capped_below_the_card_count(self) -> None:
        vectors = PpmiSvdMethod(PpmiSvdOptions(dim=64)).fit(_counts(_two_clusters()))

        assert vectors.shape == (6, 5)

    def test_is_deterministic(self) -> None:
        counts = _counts(_two_clusters())
        method = PpmiSvdMethod(PpmiSvdOptions(dim=3))

        assert np.array_equal(method.fit(counts), method.fit(counts))

    def test_card_with_no_pair_gets_a_zero_row(self) -> None:
        pairs = np.zeros((7, 7))
        pairs[:6, :6] = _two_clusters()

        vectors = PpmiSvdMethod(PpmiSvdOptions(dim=3)).fit(_counts(pairs))

        assert np.allclose(vectors[6], 0.0)

    def test_one_card_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 cards"):
            PpmiSvdMethod(PpmiSvdOptions(dim=3)).fit(_counts(np.zeros((1, 1))))

    def test_no_pair_raises(self) -> None:
        with pytest.raises(ValueError, match="no two cards"):
            PpmiSvdMethod(PpmiSvdOptions(dim=3)).fit(_counts(np.zeros((3, 3))))


class TestTeacherMethodFor:
    def test_ppmi_svd_options_pick_ppmi_svd(self) -> None:
        method = teacher_method_for(PpmiSvdOptions(dim=2))

        assert method.fit(_counts(_two_clusters())).shape == (6, 2)


class TestPpmiSvdOptions:
    @pytest.mark.parametrize(
        "options",
        [
            {"dim": 0},
            {"dim": 4, "context_smoothing": 0.0},
            {"dim": 4, "context_smoothing": 1.5},
        ],
    )
    def test_rejects_invalid_options(self, options: dict) -> None:
        with pytest.raises(ValueError):
            PpmiSvdOptions(**options)

    def test_payload(self) -> None:
        assert PpmiSvdOptions(dim=64).payload() == {
            "kind": "ppmi_svd",
            "dim": 64,
            "context_smoothing": 0.75,
        }


class TestPpmiValues:
    def test_matches_a_hand_computed_ppmi(self) -> None:
        pairs = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        alpha = 0.75

        ppmi = _ppmi_of(scipy.sparse.csr_matrix(pairs), alpha).toarray()

        total = pairs.sum()
        rows = pairs.sum(axis=1)
        context = rows**alpha / (rows**alpha).sum()
        expected = np.zeros_like(pairs)
        for i in range(3):
            for j in range(3):
                if pairs[i, j] > 0:
                    pmi = np.log(
                        (pairs[i, j] / total) / ((rows[i] / total) * context[j])
                    )
                    expected[i, j] = max(pmi, 0.0)
        assert np.allclose(ppmi, expected)
