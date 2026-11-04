"""summarize_deck_scores: the scalars DeckRetrievalAuc writes, from its
scored decks. Private to DeckRetrievalAuc (single consumer).

Every mean is unweighted at each level: anchors within a deck (DeckScore),
decks within a game, games overall - so a game with many sampled decks
does not outweigh one with few.
"""

from typing import Callable, Sequence

import numpy as np

from src.evaluation.intrinsic.deck_retrieval.deck_score import DeckScore
from src.schema.game_id import GameId

_OVERALL_KEY = "deck_auc"
_LOWER_PERCENTILE = 2.5
_UPPER_PERCENTILE = 97.5


def summarize_deck_scores(
    scores: Sequence[DeckScore], bootstrap_draws: int, seed: np.random.SeedSequence
) -> dict[str, float]:
    """The AUC scalars: overall and per game, each with a 95% percentile
    bootstrap interval (decks resampled with replacement within each game,
    the overall mean recomputed from each draw's game means), plus the
    per-tier means. The interval covers which decks were sampled, for this
    encoder's fixed card vectors; decks share negatives, so it is a little
    narrow, and it says nothing about retraining noise.

    Inputs: scores (non-empty), bootstrap_draws (>= 1), seed (seeds the
        bootstrap's generator).
    Output: {"deck_auc", "deck_auc_lo95", "deck_auc_hi95",
        "deck_auc.<game>", "deck_auc.<game>_lo95", "deck_auc.<game>_hi95"
        per game, "deck_auc.train_anchor", "deck_auc.heldout_anchor"}; a
        tier key is left out when no deck has an anchor of that tier.
    Side effects: none.
    Exceptions: none (the caller guarantees scores is non-empty).

    Example:
        >>> summarize_deck_scores(scores, bootstrap_draws=1000, seed=SeedSequence(0))[
        ...     "deck_auc"]
        0.83
    """
    result: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    deck_aucs = _group_by_game(scores, lambda score: score.auc)

    # Each game's AUC and its bootstrap draws; the overall draws average them
    draws_by_game: dict[GameId, np.ndarray] = {}
    for game, aucs in deck_aucs.items():
        draws_by_game[game] = _bootstrap_means(aucs, bootstrap_draws, rng)
        game_key = f"{_OVERALL_KEY}.{game.value}"
        result[game_key] = float(aucs.mean())
        _add_interval(result, game_key, draws_by_game[game])
    result[_OVERALL_KEY] = _macro_mean(deck_aucs)
    _add_interval(result, _OVERALL_KEY, np.mean(list(draws_by_game.values()), axis=0))

    # The per-tier means (no interval)
    for tier_key, train in (("train_anchor", True), ("heldout_anchor", False)):
        tier_aucs = _group_by_game(scores, lambda score: score.tier_auc(train))
        if tier_aucs:
            result[f"{_OVERALL_KEY}.{tier_key}"] = _macro_mean(tier_aucs)
    return result


def _group_by_game(
    scores: Sequence[DeckScore], deck_value: Callable[[DeckScore], float | None]
) -> dict[GameId, np.ndarray]:
    """Group deck_value(score) by game, skipping None; games with no value
    are left out.

    Inputs: scores; deck_value (one number per deck, or None).
    Output: GameId -> float64 array, in first-seen game order.
    Side effects: none. Exceptions: none.
    """
    result: dict[GameId, list[float]] = {}
    for score in scores:
        value = deck_value(score)
        if value is not None:
            result.setdefault(score.source_game, []).append(value)
    return {game: np.array(values) for game, values in result.items()}


def _macro_mean(values_by_game: dict[GameId, np.ndarray]) -> float:
    """Mean over games of each game's mean.

    Inputs: values_by_game (non-empty, non-empty arrays). Output: float.
    Side effects: none. Exceptions: none.
    """
    return float(np.mean([values.mean() for values in values_by_game.values()]))


def _bootstrap_means(
    values: np.ndarray, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """draws means of values resampled with replacement.

    Inputs: values (n,), n >= 1; draws (>= 1); rng (advanced).
    Output: float64 (draws,). Side effects: advances rng. Exceptions: none.
    """
    picks = rng.integers(0, len(values), size=(draws, len(values)))
    return values[picks].mean(axis=1)


def _add_interval(scalars: dict[str, float], key: str, draw_means: np.ndarray) -> None:
    """Set scalars[key + "_lo95"] and scalars[key + "_hi95"] to the 2.5th
    and 97.5th percentiles of draw_means.

    Inputs: scalars (mutated), key, draw_means (draws,). Output: None.
    Side effects: writes two keys into scalars. Exceptions: none.
    """
    lower, upper = np.percentile(draw_means, [_LOWER_PERCENTILE, _UPPER_PERCENTILE])
    scalars[f"{key}_lo95"] = float(lower)
    scalars[f"{key}_hi95"] = float(upper)
