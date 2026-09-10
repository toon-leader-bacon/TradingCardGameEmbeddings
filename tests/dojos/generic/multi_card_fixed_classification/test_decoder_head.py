import torch

from src.dojos.generic.multi_card_fixed_classification.decoder_head import (
    MultiCardFixedClassificationDecoderHead,
)
from src.dojos.generic.pooling import MeanEmbeddingPooler


class _StubPooler:
    """Always returns a fixed vector - isolates the decoder head's own
    stack/MLP wiring from MeanEmbeddingPooler's actual math."""

    def pool(self, embeddings):
        return torch.zeros(embeddings[0].shape[0])


class TestForward:
    def test_returns_num_classes_logits_per_deck(self) -> None:
        head = MultiCardFixedClassificationDecoderHead(
            card_embedding_size=4, num_classes=3
        )
        embeddings = [
            [torch.randn(4), torch.randn(4)],  # a 2-card deck
            [torch.randn(4)],  # a 1-card deck - decks vary in size
        ]

        output = head(embeddings)

        assert output.shape == (2, 3)

    def test_uses_injected_pooler_by_default_mean(self) -> None:
        default_head = MultiCardFixedClassificationDecoderHead(
            card_embedding_size=4, num_classes=3
        )
        stub_head = MultiCardFixedClassificationDecoderHead(
            card_embedding_size=4, num_classes=3, pooler=_StubPooler()
        )
        assert isinstance(default_head.pooler, MeanEmbeddingPooler)
        assert isinstance(stub_head.pooler, _StubPooler)


class TestForwardSingle:
    def test_returns_num_classes_vector(self) -> None:
        head = MultiCardFixedClassificationDecoderHead(
            card_embedding_size=4, num_classes=3
        )

        result = head.forward_single([torch.randn(4), torch.randn(4)])

        assert result.shape == (3,)
