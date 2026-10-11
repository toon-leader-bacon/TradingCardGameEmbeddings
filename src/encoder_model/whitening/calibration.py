"""Fitting a token whitening: which cards to fit on, and the fit itself.

Two steps, kept apart so each can be reused: sample_calibration_cards picks
TRAIN-tier cards from a card lookup and returns them with the holdout that
picked them (a CalibrationSample); fit_token_whitening encodes the sample
and turns its token statistics into WhiteningStats, recording that same
holdout. See scripts/fit_whitening.py for the command line.
"""

import random
from dataclasses import dataclass

import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.encoder_model.card_serialization import serialize_card_to_json
from src.encoder_model.text_encoder import PretrainedTextEncoder, TokenEncoding
from src.encoder_model.whitening.covariance_accumulator import CovarianceAccumulator
from src.encoder_model.whitening.whitening_stats import WhiteningStats
from src.schema.card import GenericCard
from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec

# Keeps the whitening finite even when every eigenvalue is 0
_MIN_EIGENVALUE_FLOOR = 1e-12


@dataclass(frozen=True)
class CalibrationSampleOptions:
    """Which cards to fit a whitening on.

    holdout: only its TRAIN-tier cards are sampled
        (HoldoutSpec.no_holdout() to sample from every card). Held-out
        games are always VALIDATION, so they are never sampled.
    games: the games to sample from, each at most cards_per_game cards.
    cards_per_game: >= 1; a cap, so a large game (MTG) does not dominate.
    seed: for the sample.
    """

    holdout: HoldoutSpec
    games: tuple[GameId, ...]
    cards_per_game: int
    seed: int = 0

    def __post_init__(self) -> None:
        """Exceptions: ValueError if games is empty or cards_per_game < 1."""
        if not self.games or self.cards_per_game < 1:
            raise ValueError(
                f"need at least one game and cards_per_game >= 1, got "
                f"{len(self.games)} games, cards_per_game={self.cards_per_game}"
            )


@dataclass(frozen=True)
class CalibrationSample:
    """Cards to fit on, and the holdout that picked them: only
    sample_calibration_cards builds one, so the holdout a fit records is
    always the one that did the sampling."""

    cards: tuple[GenericCard, ...]
    holdout: HoldoutSpec


@dataclass(frozen=True)
class WhiteningFitOptions:
    """How to fit a whitening to a sample.

    components: dimensions to keep (the top principal components); None
        keeps every dimension of the text encoder.
    batch_size: cards per text-encoder call.
    relative_eps: added to every eigenvalue, relative to the largest, so
        near-zero directions are not blown up into noise.
    """

    components: int | None = None
    batch_size: int = 32
    relative_eps: float = 1e-5

    def __post_init__(self) -> None:
        """Exceptions: ValueError if components (when given) or batch_size
        < 1, or relative_eps <= 0."""
        if self.components is not None and self.components < 1:
            raise ValueError(f"components must be >= 1, got {self.components}")
        if self.batch_size < 1 or self.relative_eps <= 0:
            raise ValueError(
                f"batch_size must be >= 1 and relative_eps > 0, got "
                f"{self.batch_size}, {self.relative_eps}"
            )


def sample_calibration_cards(
    card_lookup: CardLookup, options: CalibrationSampleOptions
) -> CalibrationSample:
    """Up to options.cards_per_game TRAIN-tier cards from each of
    options.games, seeded; each game's unknown-card placeholder excluded.

    Inputs: card_lookup (every game in options.games), options.
    Output: CalibrationSample, cards grouped by game in options.games
        order, holdout = options.holdout.
    Side effects: none (reads card_lookup).
    Exceptions: ValueError if a game has no TRAIN-tier card (e.g. a
        held-out game was listed).

    Example:
        >>> sample = sample_calibration_cards(lookup, options)
        >>> len(sample.cards) <= options.cards_per_game * len(options.games)
        True
    """
    cards: list[GenericCard] = []
    rng = random.Random(options.seed)

    # Each game in turn: its TRAIN-tier cards, then a capped sample
    for game in options.games:
        candidates = _train_tier_cards(card_lookup, game, options.holdout)
        if not candidates:
            raise ValueError(
                f"no TRAIN-tier {game.value} card to calibrate on; "
                "is it a held-out game?"
            )
        count = min(options.cards_per_game, len(candidates))
        cards.extend(rng.sample(candidates, count))
    return CalibrationSample(cards=tuple(cards), holdout=options.holdout)


def fit_token_whitening(
    text_encoder: PretrainedTextEncoder,
    sample: CalibrationSample,
    options: WhiteningFitOptions,
) -> WhiteningStats:
    """Encode the sample's cards (serialized exactly as
    CardEmbeddingPipeline does), and fit a PCA whitening to every real
    token's state.

    Inputs: text_encoder (frozen; its checkpoint is recorded), sample
        (its holdout is recorded), options.
    Output: WhiteningStats, input_dim = text_encoder.output_dim,
        output_dim = options.components or all of it.
    Side effects: runs the text encoder under no_grad on its device.
    Exceptions: ValueError if the sample has no cards or options.components
        > text_encoder.output_dim.

    Example:
        >>> stats = fit_token_whitening(encoder, sample, WhiteningFitOptions())
        >>> stats.input_dim
        768
    """
    # Validate inputs
    components = options.components or text_encoder.output_dim
    if not sample.cards or components > text_encoder.output_dim:
        raise ValueError(
            f"need cards and components <= {text_encoder.output_dim}; "
            f"got {len(sample.cards)} cards, components={options.components}"
        )

    # Accumulate every real token of every card, one batch at a time
    accumulator = CovarianceAccumulator(text_encoder.output_dim)
    with torch.no_grad():
        for start in range(0, len(sample.cards), options.batch_size):
            batch = sample.cards[start : start + options.batch_size]
            encoding = text_encoder.encode([serialize_card_to_json(c) for c in batch])
            accumulator.add(_real_tokens(encoding))

    # PCA whitening of the token distribution
    projection = _whitening_projection(
        accumulator.covariance(), components, options.relative_eps
    )
    return WhiteningStats(
        checkpoint=text_encoder.checkpoint,
        holdout=sample.holdout,
        card_count=len(sample.cards),
        mean=accumulator.mean().float(),
        projection=projection.float(),
    )


def _train_tier_cards(
    card_lookup: CardLookup, game: GameId, holdout: HoldoutSpec
) -> list[GenericCard]:
    """Private helper - single consumer is sample_calibration_cards.

    Inputs: card_lookup, game, holdout.
    Output: game's cards that holdout puts in CardTier.TRAIN, without the
        game's unknown-card placeholder (CardBinder.unknown_card_uuid).
    Side effects: none. Exceptions: none.
    """
    unknown = CardBinder.unknown_card_uuid(game)
    return [
        card
        for card in card_lookup.all_cards(game)
        if card.nocab_uuid != unknown
        and holdout.tier_of(card.nocab_uuid, game) is CardTier.TRAIN
    ]


def _real_tokens(encoding: TokenEncoding) -> torch.Tensor:
    """Private helper - single consumer is fit_token_whitening.

    Inputs: encoding. Output: (n_real_tokens, hidden_dim), padding dropped.
    Side effects: none. Exceptions: none.
    """
    return encoding.hidden_states[encoding.attention_mask.bool()]


def _whitening_projection(
    covariance: torch.Tensor, components: int, relative_eps: float
) -> torch.Tensor:
    """Private helper - single consumer is fit_token_whitening.

    Inputs: covariance (dim, dim) float64, components (<= dim),
        relative_eps.
    Output: (dim, components) float64: the top-components eigenvectors,
        each divided by sqrt(eigenvalue + relative_eps * largest
        eigenvalue), so projected tokens have unit variance per component.
    Side effects: none. Exceptions: none.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    # eigh sorts ascending; keep the largest components, largest first
    order = eigenvalues.argsort(descending=True)[:components]
    # The absolute minimum only matters for degenerate (constant) tokens
    floor = (relative_eps * eigenvalues.max()).clamp(min=_MIN_EIGENVALUE_FLOOR)
    scale = torch.sqrt(eigenvalues[order].clamp(min=0) + floor)
    return eigenvectors[:, order] / scale
