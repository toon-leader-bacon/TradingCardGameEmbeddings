import torch

from src.dojos.generic.multi_card_regression.decoder_head import (
    MultiCardRegressionDecoderHead,
)
from src.dojos.generic.pooling import MeanEmbeddingPooler


class _StubPooler:
    """Always returns a fixed vector - isolates the decoder head's own
    stack/MLP/squeeze wiring from MeanEmbeddingPooler's actual math."""

    def pool(self, embeddings):
        return torch.zeros(embeddings[0].shape[0])


class TestMeanEmbeddingPooler:
    def test_pools_via_unweighted_mean(self) -> None:
        pooler = MeanEmbeddingPooler()

        result = pooler.pool([torch.ones(4), torch.zeros(4)])

        assert torch.allclose(result, torch.full((4,), 0.5))

    def test_single_embedding_deck_returns_it_unchanged(self) -> None:
        pooler = MeanEmbeddingPooler()
        embedding = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = pooler.pool([embedding])

        assert torch.allclose(result, embedding)


class TestForward:
    def test_returns_one_prediction_per_deck(self) -> None:
        head = MultiCardRegressionDecoderHead(card_embedding_size=4)
        embeddings = [
            [torch.randn(4), torch.randn(4)],  # a 2-card deck
            [torch.randn(4)],  # a 1-card deck - decks vary in size
        ]

        output = head(embeddings)

        assert output.shape == (2,)

    def test_uses_injected_pooler_by_default_mean(self) -> None:
        default_head = MultiCardRegressionDecoderHead(card_embedding_size=4)
        stub_head = MultiCardRegressionDecoderHead(
            card_embedding_size=4, pooler=_StubPooler()
        )
        assert isinstance(default_head.pooler, MeanEmbeddingPooler)
        assert isinstance(stub_head.pooler, _StubPooler)


class TestForwardSingle:
    def test_returns_scalar(self) -> None:
        head = MultiCardRegressionDecoderHead(card_embedding_size=4)

        result = head.forward_single([torch.randn(4), torch.randn(4)])

        assert result.shape == ()
