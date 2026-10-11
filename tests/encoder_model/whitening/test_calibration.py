"""Whitening calibration: TRAIN-tier sampling and the token fit."""

from datetime import datetime, timezone
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.encoder_model.card_serialization import serialize_card_to_json
from src.encoder_model.text_encoder import PretrainedTextEncoder
from src.encoder_model.whitening.calibration import (
    CalibrationSample,
    CalibrationSampleOptions,
    WhiteningFitOptions,
    fit_token_whitening,
    sample_calibration_cards,
)
from src.encoder_model.whitening.whitening_stats import WhiteningStats
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec
from tests.encoder_model.fakes import HIDDEN, FakeTextEncoder

HOLDOUT = HoldoutSpec(
    seed=0, tier_ratios=(8, 1, 1), held_out_games=frozenset({GameId.GWENT})
)


def _card(name: str, game: GameId, nocab_uuid: UUID | None = None) -> GenericCard:
    return GenericCard(
        nocab_uuid=nocab_uuid or uuid4(),
        source_game=game,
        name=name,
        raw_content={"name": name, "text": f"rules text of {name}"},
        provenance=Provenance(
            DataSource.SCRYFALL, name, datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


class _Lookup:
    """all_cards over a fixed dict: the only CardLookup method used."""

    def __init__(self, cards: dict[GameId, list[GenericCard]]) -> None:
        self._cards = cards

    def all_cards(self, source_game: GameId) -> list[GenericCard]:
        return list(self._cards[source_game])


def _lookup() -> CardLookup:
    cards = {
        game: [_card(f"{game.value} {i}", game) for i in range(60)]
        for game in (GameId.MTG, GameId.DOMINION, GameId.GWENT)
    }
    unknown = CardBinder.unknown_card_uuid(GameId.MTG)
    cards[GameId.MTG].append(_card("unknown", GameId.MTG, unknown))
    return cast(CardLookup, _Lookup(cards))


class _FakePretrainedEncoder(FakeTextEncoder):
    """FakeTextEncoder with the checkpoint name fit_token_whitening records."""

    checkpoint = "fake-ckpt"


def _sample_options(**overrides: Any) -> CalibrationSampleOptions:
    values: dict[str, Any] = {
        "holdout": HOLDOUT,
        "games": (GameId.MTG, GameId.DOMINION),
        "cards_per_game": 10,
    }
    values.update(overrides)
    return CalibrationSampleOptions(**values)


def test_the_sample_is_train_tier_only_capped_and_carries_its_holdout() -> None:
    sample = sample_calibration_cards(_lookup(), _sample_options())
    assert sample.holdout == HOLDOUT
    assert len(sample.cards) == 20
    for card in sample.cards:
        assert HOLDOUT.tier_of(card.nocab_uuid, card.source_game) is CardTier.TRAIN
        assert card.nocab_uuid != CardBinder.unknown_card_uuid(card.source_game)


def test_a_cap_above_a_game_size_takes_all_its_train_cards() -> None:
    sample = sample_calibration_cards(_lookup(), _sample_options(cards_per_game=1000))
    mtg = [card for card in sample.cards if card.source_game is GameId.MTG]
    # Some of the 60 fall in TEST/VALIDATION; the placeholder never counts
    assert 0 < len(mtg) < 60


def test_the_sample_is_seeded() -> None:
    lookup = _lookup()
    first = sample_calibration_cards(lookup, _sample_options())
    again = sample_calibration_cards(lookup, _sample_options())
    other = sample_calibration_cards(lookup, _sample_options(seed=1))
    assert [c.nocab_uuid for c in first.cards] == [c.nocab_uuid for c in again.cards]
    assert [c.nocab_uuid for c in first.cards] != [c.nocab_uuid for c in other.cards]


def test_a_held_out_game_cannot_be_sampled() -> None:
    with pytest.raises(ValueError, match="gwent"):
        sample_calibration_cards(_lookup(), _sample_options(games=(GameId.GWENT,)))


@pytest.mark.parametrize("overrides", [{"games": ()}, {"cards_per_game": 0}])
def test_bad_sample_options_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _sample_options(**overrides)


@pytest.mark.parametrize(
    "overrides", [{"components": 0}, {"batch_size": 0}, {"relative_eps": 0.0}]
)
def test_bad_fit_options_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        WhiteningFitOptions(**overrides)


def _fit(components: int | None = None) -> tuple[WhiteningStats, CalibrationSample]:
    sample = sample_calibration_cards(_lookup(), _sample_options(cards_per_game=40))
    encoder = cast(PretrainedTextEncoder, _FakePretrainedEncoder())
    options = WhiteningFitOptions(
        components=components, batch_size=7, relative_eps=1e-9
    )
    return fit_token_whitening(encoder, sample, options), sample


def test_the_fit_records_where_it_came_from() -> None:
    stats, sample = _fit()
    assert stats.checkpoint == "fake-ckpt"
    assert stats.holdout == sample.holdout
    assert stats.card_count == len(sample.cards)
    assert (stats.input_dim, stats.output_dim) == (HIDDEN, HIDDEN)


def test_whitened_sample_tokens_have_unit_covariance() -> None:
    stats, sample = _fit()
    texts = [serialize_card_to_json(card) for card in sample.cards]
    encoding = FakeTextEncoder().encode(texts)
    tokens = encoding.hidden_states[encoding.attention_mask.bool()].double()
    whitened = (tokens - stats.mean.double()) @ stats.projection.double()
    covariance = torch.cov(whitened.T, correction=0)
    identity = torch.eye(HIDDEN, dtype=torch.float64)
    assert torch.allclose(covariance, identity, atol=1e-3)


def test_components_keep_the_top_directions_largest_first() -> None:
    stats, sample = _fit(components=3)
    texts = [serialize_card_to_json(card) for card in sample.cards]
    encoding = FakeTextEncoder().encode(texts)
    tokens = encoding.hidden_states[encoding.attention_mask.bool()].double()
    eigenvalues, eigenvectors = torch.linalg.eigh(torch.cov(tokens.T, correction=0))
    top = eigenvectors[:, eigenvalues.argsort(descending=True)[:3]]
    # Each kept column points along the matching top eigenvector, in order
    columns = torch.nn.functional.normalize(stats.projection.double(), dim=0)
    alignment = (columns * top).sum(dim=0).abs()
    assert torch.allclose(alignment, torch.ones(3, dtype=torch.float64), atol=1e-4)
    # And the sample's tokens, projected onto them, are white
    whitened = (tokens - stats.mean.double()) @ stats.projection.double()
    identity = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(torch.cov(whitened.T, correction=0), identity, atol=1e-3)


def test_more_components_than_the_encoder_has_are_refused() -> None:
    with pytest.raises(ValueError, match="components"):
        _fit(components=HIDDEN + 1)
