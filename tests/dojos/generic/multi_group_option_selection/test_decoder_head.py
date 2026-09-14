import torch

from src.dojos.generic.multi_group_option_selection.decoder_head import (
    MultiGroupOptionSelectionDecoderHead,
)
from src.dojos.generic.option_scoring import BilinearOptionScoringHead
from src.dojos.generic.pooling import MeanEmbeddingPooler


class _RecordingScoringHead:
    """Records the context each call was given, returns a fixed-length
    zero vector - isolates the decoder head's own pooling/iteration
    from BilinearOptionScoringHead's actual math."""

    def __init__(self) -> None:
        self.received_contexts: list = []

    def score(self, option_embeddings, context):
        self.received_contexts.append(context)
        return torch.zeros(len(option_embeddings))


class TestForwardSingle:
    def test_empty_pool_scores_with_no_context(self) -> None:
        scoring_head = _RecordingScoringHead()
        head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size=4, scoring_head=scoring_head
        )
        option_embeddings = [torch.randn(4), torch.randn(4)]

        result = head.forward_single([option_embeddings, []])

        assert result.shape == (2,)
        assert scoring_head.received_contexts == [None]

    def test_non_empty_pool_scores_with_a_pooled_context(self) -> None:
        scoring_head = _RecordingScoringHead()
        head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size=4, scoring_head=scoring_head
        )
        option_embeddings = [torch.randn(4)]
        pool_embeddings = [torch.randn(4), torch.randn(4)]

        head.forward_single([option_embeddings, pool_embeddings])

        assert len(scoring_head.received_contexts) == 1
        assert scoring_head.received_contexts[0] is not None
        assert scoring_head.received_contexts[0].shape == (4,)


class TestForward:
    def test_returns_one_ragged_logits_tensor_per_example(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(card_embedding_size=4)
        embeddings = [
            [[torch.randn(4), torch.randn(4)], []],  # 2 options, no pool
            [[torch.randn(4)], [torch.randn(4)]],  # 1 option, 1-card pool
        ]

        output = head(embeddings)

        assert [logits.shape[0] for logits in output] == [2, 1]

    def test_uses_injected_collaborators_by_default(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(card_embedding_size=4)

        assert isinstance(head.scoring_head, BilinearOptionScoringHead)
        assert isinstance(head.pooler, MeanEmbeddingPooler)
