import torch

from src.dojos.generic.multi_group_regression.decoder_head import (
    MultiGroupRegressionDecoderHead,
)
from src.dojos.generic.pooling import MeanEmbeddingPooler


class _StubPooler:
    """Always returns a fixed vector - isolates the decoder head's own
    stack/MLP wiring from MeanEmbeddingPooler's actual math."""

    def pool(self, embeddings):
        return torch.zeros(embeddings[0].shape[0])


class TestForward:
    def test_returns_one_scalar_per_example(self) -> None:
        head = MultiGroupRegressionDecoderHead(card_embedding_size=4)
        embeddings = [
            [[torch.randn(4)], [torch.randn(4), torch.randn(4)]],
            [[torch.randn(4), torch.randn(4)], []],  # empty second group
        ]

        output = head(embeddings)

        assert output.shape == (2,)

    def test_uses_injected_pooler_by_default_mean(self) -> None:
        default_head = MultiGroupRegressionDecoderHead(card_embedding_size=4)
        stub_head = MultiGroupRegressionDecoderHead(
            card_embedding_size=4, pooler=_StubPooler()
        )
        assert isinstance(default_head.pooler, MeanEmbeddingPooler)
        assert isinstance(stub_head.pooler, _StubPooler)


class TestForwardSingle:
    def test_returns_scalar(self) -> None:
        head = MultiGroupRegressionDecoderHead(card_embedding_size=4)

        result = head.forward_single([[torch.randn(4)], [torch.randn(4)]])

        assert result.shape == ()

    def test_empty_second_group_does_not_raise(self) -> None:
        head = MultiGroupRegressionDecoderHead(card_embedding_size=4)

        result = head.forward_single([[torch.randn(4), torch.randn(4)], []])

        assert result.shape == ()

    def test_empty_second_group_uses_the_learned_placeholder(self) -> None:
        head = MultiGroupRegressionDecoderHead(card_embedding_size=4)
        group_0 = [torch.randn(4)]

        with_empty_group_1 = head.forward_single([group_0, []])
        with_nonempty_group_1 = head.forward_single([group_0, [torch.randn(4)]])

        # Different group-1 content must change the prediction - the
        # placeholder isn't silently ignored.
        assert with_empty_group_1.item() != with_nonempty_group_1.item()


class TestEmptySecondGroupParameter:
    def test_is_a_registered_parameter(self) -> None:
        head = MultiGroupRegressionDecoderHead(card_embedding_size=4)

        assert any(p is head.empty_second_group for p in head.parameters())

    def test_receives_a_gradient_when_group_1_is_empty(self) -> None:
        head = MultiGroupRegressionDecoderHead(card_embedding_size=4)

        result = head.forward_single([[torch.randn(4)], []])
        result.backward()

        assert head.empty_second_group.grad is not None
