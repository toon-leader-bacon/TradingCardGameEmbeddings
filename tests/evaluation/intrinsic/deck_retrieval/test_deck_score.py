from uuid import uuid4

import numpy as np
import pytest

from src.evaluation.intrinsic.deck_retrieval.deck_score import DeckScore, anchor_aucs
from src.schema.game_id import GameId


def _brute_force_aucs(members: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    """The Mann-Whitney definition, pair by pair."""
    result = []
    for anchor in range(len(members)):
        total = 0.0
        pairs = 0
        for mate in range(len(members)):
            if mate == anchor:
                continue
            mate_cosine = members[anchor] @ members[mate]
            for negative in negatives:
                negative_cosine = members[anchor] @ negative
                total += 1.0 if mate_cosine > negative_cosine else 0.0
                total += 0.5 if mate_cosine == negative_cosine else 0.0
                pairs += 1
        result.append(total / pairs)
    return np.array(result)


def _aucs(members: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    """anchor_aucs over a game holding members then negatives."""
    game = np.vstack([members, negatives]).astype(np.float64)
    member_rows = np.arange(len(members))
    negative_rows = np.arange(len(members), len(game))
    return anchor_aucs(game[member_rows] @ game.T, member_rows, negative_rows)


def _score(aucs: list[float], is_train: list[bool]) -> DeckScore:
    return DeckScore(uuid4(), GameId.MTG, np.array(aucs), np.array(is_train))


class TestAnchorAucs:
    def test_deck_mates_always_closer_scores_one(self) -> None:
        members = np.array([[1.0, 0.0], [0.99, 0.141], [0.98, 0.199]])
        negatives = np.array([[0.0, 1.0], [-1.0, 0.0]])
        assert np.allclose(_aucs(members, negatives), 1.0)

    def test_negatives_always_closer_scores_zero(self) -> None:
        members = np.array([[1.0, 0.0], [-1.0, 0.0]])
        negatives = np.array([[0.0, 1.0], [0.0, -1.0]])
        assert np.allclose(_aucs(members, negatives), 0.0)

    def test_all_ties_score_one_half(self) -> None:
        vector = np.array([[0.6, 0.8]])
        members = np.repeat(vector, 3, axis=0)
        negatives = np.repeat(vector, 4, axis=0)
        assert np.allclose(_aucs(members, negatives), 0.5)

    def test_hand_computed_partial_score(self) -> None:
        # Anchor 0 = e0: mates e0 (cos 1) and e1 (cos 0); negatives with
        # cos 0.5 and 0 -> wins 1, 1, 0, tie -> 2.5 / 4
        members = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        negatives = np.array([[0.5, 0.0, 0.866], [0.0, 0.0, 1.0]])
        assert _aucs(members, negatives)[0] == pytest.approx(2.5 / 4)

    def test_matches_brute_force_with_ties(self) -> None:
        # Small-integer vectors: every cosine is exact, so ties are real ties
        rng = np.random.default_rng(0)
        members = rng.integers(0, 3, size=(6, 4)).astype(np.float32)
        negatives = np.vstack([rng.integers(0, 3, size=(20, 4)), members[:3]]).astype(
            np.float32
        )
        assert np.allclose(
            _aucs(members, negatives), _brute_force_aucs(members, negatives)
        )

    def test_columns_outside_both_row_sets_are_ignored(self) -> None:
        # Column 3 (a negative on top of the deck) is neither member nor
        # negative, so it cannot lower the score
        game = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0], [1.0, 0.0]])
        members, negatives = np.array([0, 1]), np.array([2])
        result = anchor_aucs(game[members] @ game.T, members, negatives)
        assert np.allclose(result, 1.0)

    def test_returns_float64_per_member(self) -> None:
        rng = np.random.default_rng(1)
        result = _aucs(
            rng.standard_normal((5, 3)).astype(np.float32),
            rng.standard_normal((7, 3)).astype(np.float32),
        )
        assert result.shape == (5,)
        assert result.dtype == np.float64
        assert ((result >= 0) & (result <= 1)).all()


class TestDeckScore:
    def test_counts_and_means(self) -> None:
        score = _score([1.0, 0.5, 0.0], [True, False, True])
        assert score.n_cards == 3
        assert score.n_train_anchors == 2
        assert score.n_heldout_anchors == 1
        assert score.auc == pytest.approx(0.5)
        assert score.tier_auc(train=True) == pytest.approx(0.5)
        assert score.tier_auc(train=False) == pytest.approx(0.5)

    def test_tier_with_no_anchor_is_none(self) -> None:
        score = _score([0.9, 0.7], [True, True])
        assert score.tier_auc(train=False) is None
        assert score.tier_auc(train=True) == pytest.approx(0.8)

    def test_rejects_misaligned_arrays(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            _score([0.5, 0.5, 0.5], [True, False])

    def test_rejects_two_dimensional_arrays(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            DeckScore(uuid4(), GameId.MTG, np.zeros((2, 2)), np.zeros((2, 2), bool))

    def test_rejects_fewer_than_two_anchors(self) -> None:
        with pytest.raises(ValueError, match=">= 2"):
            _score([0.5], [True])
