"""A CardEmbeddingPipeline with a PooledFeatureCache: the same embeddings as
without one, the frozen stages skipped on hits, warm() filling ahead, the
state_dict untouched, and trainable stages before the cut refused. Also the
models' warm()/feature_cache_stats() entry points."""

from datetime import datetime, timezone
from typing import List
from uuid import uuid4

import pytest
import torch

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.embedding_head import LinearEmbeddingHead
from src.encoder_model.feature_cache.pooled_feature_cache import PooledFeatureCache
from src.encoder_model.multi_card_model import MultiCardModel
from src.encoder_model.pooling.attention_pooling import AttentionPooling
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.encoder_model.single_card_model import SingleCardModel
from src.encoder_model.text_encoder import TokenEncoding
from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.encoder_model.fakes import HIDDEN, FakeTextEncoder
from tests.encoder_model.feature_cache.cache_fakes import memory_cache

_EMBED = 4


class _CountingTextEncoder(FakeTextEncoder):
    """FakeTextEncoder that records every batch of texts it encodes."""

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []

    def encode(self, texts: List[str]) -> TokenEncoding:
        self.batches.append(list(texts))
        return super().encode(texts)


def _card(name: str, text: str = "") -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={"name": name, "text": text or f"rules for {name}"},
        provenance=Provenance(
            DataSource.SCRYFALL, name, datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


def _cache(capacity: int = 100) -> PooledFeatureCache:
    return memory_cache(HIDDEN, capacity)


def _pipelines() -> tuple[CardEmbeddingPipeline, CardEmbeddingPipeline]:
    """(live, cached) pipelines sharing one head, so outputs compare."""
    torch.manual_seed(0)
    head = LinearEmbeddingHead(HIDDEN, _EMBED)
    live = CardEmbeddingPipeline(FakeTextEncoder(), None, MeanPooling(HIDDEN), head)
    cached = CardEmbeddingPipeline(
        _CountingTextEncoder(), None, MeanPooling(HIDDEN), head, _cache()
    )
    return live, cached


CARDS = [_card("Alpha"), _card("Bravo"), _card("Charlie")]


def test_cached_embed_equals_live_embed_on_misses_and_hits() -> None:
    live, cached = _pipelines()
    expected = live.embed(CARDS)
    assert torch.equal(cached.embed(CARDS), expected)  # all misses
    assert torch.equal(cached.embed(CARDS), expected)  # all hits
    mixed = [CARDS[2], _card("Delta"), CARDS[0]]
    assert torch.equal(cached.embed(mixed), live.embed(mixed))


def test_hits_skip_the_text_encoder() -> None:
    _, cached = _pipelines()
    encoder = cached.text_encoder
    assert isinstance(encoder, _CountingTextEncoder)
    cached.embed(CARDS)
    cached.embed(CARDS[:2])
    assert len(encoder.batches) == 1


def test_a_modded_card_keeps_its_uuid_but_gets_its_own_features() -> None:
    # The prototype keyed by uuid and served unmodded features to modded cards
    live, cached = _pipelines()
    original = CARDS[0]
    modded = GenericCard(
        nocab_uuid=original.nocab_uuid,
        source_game=original.source_game,
        name=original.name,
        raw_content={"name": original.name, "text": "[MASK]"},
        provenance=original.provenance,
    )
    cached.embed([original])
    assert torch.equal(cached.embed([modded]), live.embed([modded]))


def test_gradients_still_reach_the_head() -> None:
    _, cached = _pipelines()
    cached.embed(CARDS)
    cached.embed(CARDS).sum().backward()
    assert cached.embedding_head.projection.weight.grad is not None


def test_the_cache_is_not_in_the_state_dict() -> None:
    live, cached = _pipelines()
    cached.embed(CARDS)
    assert list(cached.state_dict()) == list(live.state_dict())
    live.load_state_dict(cached.state_dict(), strict=True)


def test_a_trainable_stage_before_the_cut_is_refused() -> None:
    with pytest.raises(ValueError, match="pooling has trainable parameters"):
        CardEmbeddingPipeline(
            FakeTextEncoder(),
            None,
            AttentionPooling(HIDDEN),
            LinearEmbeddingHead(HIDDEN, _EMBED),
            _cache(),
        )


def test_warm_fills_the_cache_in_eval_mode_and_restores_train_mode() -> None:
    _, cached = _pipelines()
    cached.train()
    cached.warm(CARDS, "fp32")
    assert cached.training
    cache = cached.feature_cache
    assert cache is not None
    assert cache.stats().tiers[0].rows_used == len(CARDS)

    encoder = cached.text_encoder
    assert isinstance(encoder, _CountingTextEncoder)
    batches_after_warm = len(encoder.batches)
    cached.embed(CARDS)
    assert len(encoder.batches) == batches_after_warm


def test_warm_without_a_cache_does_nothing() -> None:
    live, _ = _pipelines()
    live.warm(CARDS, "fp32")
    assert live.feature_cache is None


@pytest.mark.parametrize("model_type", [SingleCardModel, MultiCardModel])
def test_models_warm_and_report_through_their_pipeline(model_type: type) -> None:
    _, cached = _pipelines()
    model = (
        SingleCardModel(cached)
        if model_type is SingleCardModel
        else MultiCardModel(cached, num_heads=2, ffn_dim=8)
    )
    model.warm(CARDS, "fp32")
    stats = model.feature_cache_stats()
    assert stats is not None
    assert stats.tiers[0].rows_used == len(CARDS)
    assert list(model.state_dict())[0].startswith("card_pipeline.")


def test_a_model_without_a_cache_reports_no_stats() -> None:
    live, _ = _pipelines()
    assert SingleCardModel(live).feature_cache_stats() is None


def test_a_multi_card_batch_with_a_cache_equals_one_without() -> None:
    live, cached = _pipelines()
    torch.manual_seed(1)
    live_model = MultiCardModel(live, num_heads=2, ffn_dim=8).eval()
    cached_model = MultiCardModel(cached, num_heads=2, ffn_dim=8).eval()
    cached_model.load_state_dict(live_model.state_dict())
    decks = [[CARDS[0], CARDS[1]], [CARDS[1], CARDS[2], CARDS[0]]]
    got, want = cached_model(decks), live_model(decks)
    assert isinstance(got, PaddedEmbeddings) and isinstance(want, PaddedEmbeddings)
    assert got.lengths == want.lengths
    assert torch.allclose(got.cards, want.cards)


def test_warm_runs_the_frozen_stages_under_the_requested_autocast() -> None:
    cached = CardEmbeddingPipeline(
        FakeTextEncoder(),
        None,
        MeanPooling(HIDDEN),
        LinearEmbeddingHead(HIDDEN, _EMBED),
        memory_cache(HIDDEN, precision="bf16"),
    )
    seen: list[bool] = []
    encode = cached.text_encoder.encode

    def recording_encode(texts: List[str]) -> TokenEncoding:
        seen.append(torch.is_autocast_enabled("cpu"))
        return encode(texts)

    cached.text_encoder.encode = recording_encode  # type: ignore[method-assign]
    cached.warm(CARDS, "bf16")
    assert seen == [True]


def test_with_a_cache_the_stages_before_the_head_stay_in_eval() -> None:
    live, cached = _pipelines()
    cached.train()
    assert cached.embedding_head.training
    assert not cached.text_encoder.training
    assert not cached.pooling.training
    live.train()
    assert live.pooling.training  # no cache: train() is the usual one
