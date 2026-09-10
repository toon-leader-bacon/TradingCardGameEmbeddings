import torch

from src.dojos.generic.single_card_fixed_classification.decoder_head import (
    FixedClassificationDecoderHead,
)


class TestForward:
    def test_returns_one_logits_row_per_embedding(self) -> None:
        head = FixedClassificationDecoderHead(card_embedding_size=4, num_classes=3)
        embeddings = [torch.randn(4), torch.randn(4), torch.randn(4)]

        result = head(embeddings)

        assert result.shape == (3, 3)

    def test_single_embedding_batch(self) -> None:
        head = FixedClassificationDecoderHead(card_embedding_size=4, num_classes=5)
        embeddings = [torch.randn(4)]

        result = head(embeddings)

        assert result.shape == (1, 5)


class TestForwardSingle:
    def test_returns_num_classes_logits(self) -> None:
        head = FixedClassificationDecoderHead(card_embedding_size=4, num_classes=5)
        embedding = torch.randn(4)

        result = head.forward_single(embedding)

        assert result.shape == (5,)
