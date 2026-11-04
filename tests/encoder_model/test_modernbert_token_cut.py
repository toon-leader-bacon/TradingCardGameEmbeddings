"""The ModernBERT layer split, the partly trainable ModernBertTextEncoder,
and a pipeline cut at token level: cached output equals uncached output.

Uses a tiny random ModernBertModel and a fake tokenizer, so nothing is
downloaded."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

import src.encoder_model.text_encoder as text_encoder_module
from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.embedding_head import LinearEmbeddingHead
from src.encoder_model.feature_cache.cut_point import TokenCutPoint
from src.encoder_model.feature_cache.feature_row_kind import TOKEN_ROWS
from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.frozen_feature_cache import FrozenFeatureCache
from src.encoder_model.feature_cache.frozen_stage_fingerprint import (
    FrozenStageFingerprint,
)
from src.encoder_model.feature_cache.slot_policy import LeastRecentlyUsed
from src.encoder_model.feature_cache.tensor_feature_slab import (
    SlabPlacement,
    TensorFeatureSlab,
)
from src.encoder_model.feature_cache.tiered_feature_store import TieredFeatureStore
from src.encoder_model.modernbert_layer_stack import ModernBertLayerStack
from src.encoder_model.modernbert_text_encoder import ModernBertTextEncoder
from src.encoder_model.pipeline_cut import (
    EncoderOutputCut,
    LayerCut,
    PooledCut,
    build_pipeline_cut,
)
from src.encoder_model.pooling.attention_pooling import AttentionPooling
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.encoder_model.feature_cache.cache_fakes import PINNED_SOURCE
from tests.encoder_model.fakes import FakeTextEncoder

HIDDEN = 32
LAYERS = 4
VOCAB = 512
CPU = torch.device("cpu")
TEXTS = ["a short text", "a somewhat longer card text here", "mid length"]


def _tiny_modernbert() -> ModernBertModel:
    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        global_attn_every_n_layers=2,
        local_attention=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    return ModernBertModel(config).eval()


class _FakeBatchEncoding(dict):
    def to(self, device: torch.device) -> "_FakeBatchEncoding":
        return _FakeBatchEncoding(
            {key: value.to(device) for key, value in self.items()}
        )


class _FakeTokenizer:
    """Token j of a text is the text's j-th character code (mod VOCAB)."""

    def __call__(
        self, texts: list, max_length: int = 64, **_: Any
    ) -> _FakeBatchEncoding:
        lengths = [min(len(text), max_length) for text in texts]
        width = max(lengths)
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros((len(texts), width), dtype=torch.long)
        for row, (text, length) in enumerate(zip(texts, lengths)):
            codes = [3 + ord(char) % (VOCAB - 3) for char in text[:length]]
            input_ids[row, :length] = torch.tensor(codes)
            mask[row, :length] = 1
        return _FakeBatchEncoding({"input_ids": input_ids, "attention_mask": mask})


@pytest.fixture
def tiny_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make PretrainedTextEncoder load the tiny model and fake tokenizer."""
    monkeypatch.setattr(
        text_encoder_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda checkpoint: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        text_encoder_module,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda checkpoint, **_: _tiny_modernbert()),
    )


def _encoder(trainable_top_layers: int = 0) -> ModernBertTextEncoder:
    return ModernBertTextEncoder(
        "tiny", trainable_top_layers, max_length=64, length_bucket_size=2
    )


def _token_cache(layer: int | None) -> FrozenFeatureCache:
    fingerprint = FrozenStageFingerprint(
        PINNED_SOURCE,
        "none",
        "none",
        "fp32",
        HIDDEN,
        FeatureStorageDtype.FP32,
        TokenCutPoint(layer),
    )
    slab = TensorFeatureSlab(
        "vram",
        10_000,
        FeatureStorageDtype.FP32,
        LeastRecentlyUsed(10_000),
        SlabPlacement.FOLLOW_FIRST_PUT,
        TOKEN_ROWS,
    )
    return FrozenFeatureCache(TieredFeatureStore([slab]), fingerprint, False)


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={"name": name, "text": f"rules for {name}"},
        provenance=Provenance(
            DataSource.SCRYFALL, name, datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


class TestModernBertLayerStack:
    @pytest.mark.parametrize("layer", range(LAYERS + 1))
    def test_the_two_halves_reproduce_the_whole_model(self, layer: int) -> None:
        model = _tiny_modernbert()
        stack = ModernBertLayerStack(model)
        ids = torch.randint(3, VOCAB, (3, 12))
        mask = torch.ones(3, 12, dtype=torch.long)
        mask[1, 7:] = 0
        mask[2, 3:] = 0
        expected = model(input_ids=ids, attention_mask=mask).last_hidden_state
        actual = stack.run_from(stack.run_through(ids, mask, layer), mask, layer)
        real = mask.unsqueeze(-1).bool()
        assert torch.allclose(
            actual[real.expand_as(actual)],
            expected[real.expand_as(expected)],
            atol=1e-5,
        )

    def test_a_texts_states_do_not_depend_on_its_batchs_padding(self) -> None:
        stack = ModernBertLayerStack(_tiny_modernbert())
        ids = torch.randint(3, VOCAB, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.long)
        mask[1, 4:] = 0
        alone = stack.run_through(ids[1:, :4], mask[1:, :4], 3)
        padded = stack.run_through(ids, mask, 3)
        assert torch.allclose(alone[0], padded[1, :4], atol=1e-5)

    def test_a_layer_out_of_range_is_refused(self) -> None:
        stack = ModernBertLayerStack(_tiny_modernbert())
        with pytest.raises(ValueError, match="layer"):
            stack.parameters_through(LAYERS + 1)

    def test_a_model_that_is_not_modernbert_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ModernBERT"):
            ModernBertLayerStack(torch.nn.Linear(2, 2))


@pytest.mark.usefixtures("tiny_checkpoint")
class TestModernBertTextEncoder:
    def test_frozen_encoding_matches_the_plain_model(self) -> None:
        encoder = _encoder()
        encoding = encoder.encode(TEXTS)
        tokenized = _FakeTokenizer()(TEXTS)
        expected = encoder.model(**tokenized).last_hidden_state
        # Real tokens only: bucketing zero-pads, the model fills padding
        real = tokenized["attention_mask"].bool()
        assert torch.equal(encoding.attention_mask.bool(), real)
        assert torch.allclose(encoding.hidden_states[real], expected[real], atol=1e-5)

    def test_only_the_top_layers_and_final_norm_train(self) -> None:
        encoder = _encoder(trainable_top_layers=1)
        assert encoder.is_frozen_through(LAYERS - 1)
        assert not encoder.is_frozen_through(LAYERS)
        trained = {name for name, p in encoder.named_parameters() if p.requires_grad}
        assert trained and all(
            name.startswith(f"model.layers.{LAYERS - 1}.")
            or name.startswith("model.final_norm")
            for name in trained
        )

    def test_frozen_layers_stay_in_eval_mode(self) -> None:
        encoder = _encoder(trainable_top_layers=1).train()
        assert not encoder.model.layers[0].training
        assert encoder.model.layers[LAYERS - 1].training

    def test_gradients_reach_only_the_trainable_layers(self) -> None:
        encoder = _encoder(trainable_top_layers=1)
        encoder.encode(TEXTS).hidden_states.sum().backward()
        top = encoder.model.layers[LAYERS - 1]
        assert all(p.grad is not None for p in top.parameters())
        assert all(p.grad is None for p in encoder.model.layers[0].parameters())

    def test_too_many_trainable_layers_are_refused(self) -> None:
        with pytest.raises(ValueError, match="trainable_top_layers"):
            _encoder(trainable_top_layers=LAYERS + 1)


@pytest.mark.usefixtures("tiny_checkpoint")
class TestTokenCuts:
    @pytest.mark.parametrize("layer", [None, *range(LAYERS + 1)])
    def test_cached_features_pool_to_the_uncached_result(
        self, layer: int | None
    ) -> None:
        encoder = _encoder()
        pooling = MeanPooling(HIDDEN)
        expected = PooledCut(encoder, None, pooling).encode_before_cut(TEXTS)
        cut = build_pipeline_cut(TokenCutPoint(layer), encoder, None, pooling)
        cache = _token_cache(layer)
        cache.features(TEXTS[:1], cut.encode_before_cut, CPU)  # one hit, two misses
        features = cache.features(TEXTS, cut.encode_before_cut, CPU)
        assert torch.allclose(cut.pool_after_cut(features), expected, atol=1e-5)

    def test_a_cut_below_a_trainable_layer_is_refused(self) -> None:
        cut = LayerCut(
            _encoder(trainable_top_layers=2), LAYERS - 1, None, MeanPooling(HIDDEN)
        )
        with pytest.raises(ValueError, match="frozen"):
            cut.check_frozen()

    def test_a_cut_inside_an_encoder_that_cannot_split_is_refused(self) -> None:
        with pytest.raises(ValueError, match="LayeredTextEncoder"):
            LayerCut(
                FakeTextEncoder(), 2, None, MeanPooling(FakeTextEncoder().output_dim)
            )

    def test_any_encoder_can_be_cut_at_its_whole_output(self) -> None:
        encoder = FakeTextEncoder()
        cut = EncoderOutputCut(encoder, None, MeanPooling(encoder.output_dim))
        cut.check_frozen()
        assert cut.encode_before_cut(TEXTS).text_count == len(TEXTS)


@pytest.mark.usefixtures("tiny_checkpoint")
class TestTokenCutPipeline:
    def test_a_token_cached_pipeline_embeds_as_the_uncached_one(self) -> None:
        encoder = _encoder()
        torch.manual_seed(1)
        head = LinearEmbeddingHead(HIDDEN, 8)
        live = CardEmbeddingPipeline(encoder, None, MeanPooling(HIDDEN), head)
        cached = CardEmbeddingPipeline(
            encoder, None, MeanPooling(HIDDEN), head, _token_cache(2)
        )
        cards = [_card("Bolt"), _card("Shock"), _card("Bolt")]
        assert torch.allclose(cached.embed(cards), live.embed(cards), atol=1e-5)
        assert torch.allclose(cached.embed(cards), live.embed(cards), atol=1e-5)

    def test_a_token_cut_trains_the_layers_and_pooling_above_it(self) -> None:
        encoder = _encoder(trainable_top_layers=1)
        pooling = AttentionPooling(HIDDEN)
        pipeline = CardEmbeddingPipeline(
            encoder,
            None,
            pooling,
            LinearEmbeddingHead(HIDDEN, 8),
            _token_cache(LAYERS - 1),
        )
        pipeline.train()
        pipeline.embed([_card("Bolt"), _card("Shock")]).sum().backward()
        assert all(p.grad is not None for p in encoder.model.layers[-1].parameters())
        assert any(p.grad is not None for p in pooling.parameters())
        assert pipeline.feature_cache is not None
        assert pipeline.feature_cache.stats().encoded == 2
