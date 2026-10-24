import torch

from src.dojos.generic.single_card_regression.decoder_head import (
    SingleCardRegressionDecoderHead,
)
from tests.dojos.padding import padded


class TestForward:
    def test_returns_one_scalar_per_embedding(self) -> None:
        head = SingleCardRegressionDecoderHead(card_embedding_size=4)
        embeddings = [torch.randn(4), torch.randn(4), torch.randn(4)]

        result = head(padded(embeddings))

        assert result.shape == (3,)

    def test_single_embedding_batch(self) -> None:
        head = SingleCardRegressionDecoderHead(card_embedding_size=4)
        embeddings = [torch.randn(4)]

        result = head(padded(embeddings))

        assert result.shape == (1,)
