import torch

from src.dojos.generic.multi_group_binary_classification.decoder_head import (
    MultiGroupBinaryClassificationDecoderHead,
)
from src.dojos.generic.pooling import MeanEmbeddingPooler


class _StubPooler:
    def pool(self, embeddings):
        return torch.zeros(embeddings[0].shape[0])


class TestForward:
    def test_returns_one_logit_per_example(self) -> None:
        head = MultiGroupBinaryClassificationDecoderHead(card_embedding_size=4)
        embeddings = [
            [[torch.randn(4)], [torch.randn(4), torch.randn(4)]],
            [[torch.randn(4), torch.randn(4)], []],
        ]

        assert head(embeddings).shape == (2,)

    def test_default_pooler_is_mean_and_injected_one_is_kept(self) -> None:
        stub = _StubPooler()
        assert isinstance(
            MultiGroupBinaryClassificationDecoderHead(4).pooler, MeanEmbeddingPooler
        )
        assert MultiGroupBinaryClassificationDecoderHead(4, pooler=stub).pooler is stub


class TestForwardSingle:
    def test_returns_scalar(self) -> None:
        head = MultiGroupBinaryClassificationDecoderHead(card_embedding_size=4)

        assert head.forward_single([[torch.randn(4)], [torch.randn(4)]]).shape == ()

    def test_group_order_matters(self) -> None:
        torch.manual_seed(0)
        head = MultiGroupBinaryClassificationDecoderHead(card_embedding_size=4)
        a, b = [torch.randn(4)], [torch.randn(4)]

        assert head.forward_single([a, b]).item() != head.forward_single([b, a]).item()

    def test_empty_second_group_uses_learned_placeholder(self) -> None:
        head = MultiGroupBinaryClassificationDecoderHead(card_embedding_size=4)

        head.forward_single([[torch.randn(4)], []]).backward()

        assert head.empty_second_group.grad is not None
        assert any(p is head.empty_second_group for p in head.parameters())
