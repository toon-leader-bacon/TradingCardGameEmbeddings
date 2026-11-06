from uuid import uuid4

import numpy as np
import pytest

from src.evaluation.intrinsic.deck_retrieval.deck_auc_summary import (
    summarize_deck_scores,
)
from src.evaluation.intrinsic.deck_retrieval.deck_score import DeckScore
from src.schema.game_id import GameId


def _score(
    aucs: list[float], is_train: list[bool] | None = None, game: GameId = GameId.MTG
) -> DeckScore:
    train = is_train if is_train is not None else [True] * len(aucs)
    return DeckScore(uuid4(), game, np.array(aucs), np.array(train))


def _summary(scores: list[DeckScore], draws: int = 200, seed: int = 0) -> dict:
    return summarize_deck_scores(scores, draws, np.random.SeedSequence(seed))


class TestSummarizeDeckScores:
    def test_docstring_example_one_game(self) -> None:
        scores = [_score([0.9, 0.7]), _score([0.6, 0.6]), _score([1.0, 1.0])]
        result = _summary(scores)
        assert result["deck_auc"] == pytest.approx(0.8)
        assert result["deck_auc.mtg"] == pytest.approx(0.8)
        assert result["deck_auc_lo95"] <= 0.8 <= result["deck_auc_hi95"]
        assert result["deck_auc.mtg_lo95"] == result["deck_auc_lo95"]

    def test_games_are_averaged_unweighted(self) -> None:
        scores = [_score([1.0, 1.0]) for _ in range(3)]
        scores.append(_score([0.5, 0.5], game=GameId.GWENT))
        result = _summary(scores)
        assert result["deck_auc"] == pytest.approx(0.75)
        assert result["deck_auc.mtg"] == pytest.approx(1.0)
        assert result["deck_auc.gwent"] == pytest.approx(0.5)

    def test_overall_interval_averages_each_games_draws(self) -> None:
        # A constant game contributes no spread: the overall draws are
        # (1 + gwent draw) / 2, not a pooled resample of every deck
        rng = np.random.default_rng(2)
        scores = [_score([1.0, 1.0]) for _ in range(5)]
        scores += [_score(list(rng.random(2)), game=GameId.GWENT) for _ in range(40)]
        result = _summary(scores, draws=300)
        assert result["deck_auc.mtg_lo95"] == pytest.approx(1.0)
        assert result["deck_auc_lo95"] == pytest.approx(
            (1.0 + result["deck_auc.gwent_lo95"]) / 2
        )
        assert result["deck_auc_hi95"] == pytest.approx(
            (1.0 + result["deck_auc.gwent_hi95"]) / 2
        )

    def test_tier_means_are_unweighted_across_games(self) -> None:
        scores = [_score([1.0, 1.0]) for _ in range(3)]
        scores.append(_score([0.0, 0.0], game=GameId.GWENT))
        assert _summary(scores)["deck_auc.train_anchor"] == pytest.approx(0.5)

    def test_single_deck_interval_collapses(self) -> None:
        result = _summary([_score([0.7, 0.9])])
        assert result["deck_auc_lo95"] == pytest.approx(0.8)
        assert result["deck_auc_hi95"] == pytest.approx(0.8)

    def test_tier_means_average_within_deck_then_across(self) -> None:
        scores = [
            _score([1.0, 0.0, 0.5], [True, False, False]),
            _score([0.8, 0.6], [True, True]),
        ]
        result = _summary(scores)
        assert result["deck_auc.train_anchor"] == pytest.approx((1.0 + 0.7) / 2)
        # Only the first deck has held-out anchors
        assert result["deck_auc.heldout_anchor"] == pytest.approx(0.25)

    def test_tier_key_absent_without_anchors_of_that_tier(self) -> None:
        result = _summary([_score([0.8, 0.6], [True, True])])
        assert "deck_auc.heldout_anchor" not in result
        assert "deck_auc.train_anchor" in result

    def test_same_seed_same_interval(self) -> None:
        rng = np.random.default_rng(0)
        scores = [_score(list(rng.random(3))) for _ in range(30)]
        assert _summary(scores, seed=4) == _summary(scores, seed=4)
        assert _summary(scores, seed=4) != _summary(scores, seed=5)

    def test_interval_brackets_the_mean(self) -> None:
        rng = np.random.default_rng(1)
        scores = [_score(list(rng.random(3))) for _ in range(50)]
        result = _summary(scores, draws=500)
        assert result["deck_auc_lo95"] < result["deck_auc"] < result["deck_auc_hi95"]
