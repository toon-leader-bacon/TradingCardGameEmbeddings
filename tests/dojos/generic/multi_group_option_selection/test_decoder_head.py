import torch

from src.dojos.generic.multi_group_option_selection.decoder_head import (
    MultiGroupOptionSelectionDecoderHead,
)
from src.dojos.generic.option_scoring import BilinearOptionScoringHead
from src.dojos.generic.pooling import MeanEmbeddingPooler


class _RecordingScoringHead:
    """Records the context each call was given, returns zero logits -
    isolates the decoder head's own pooling from
    BilinearOptionScoringHead's actual math."""

    def __init__(self) -> None:
        self.received_contexts: list = []

    def score(self, options, mask, context, context_mask):
        self.received_contexts.append((context, context_mask))
        return torch.zeros(mask.shape)


class TestForward:
    def test_an_empty_pool_is_marked_as_no_context(self) -> None:
        scoring_head = _RecordingScoringHead()
        head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size=4, scoring_head=scoring_head
        )
        pool = [torch.randn(4), torch.randn(4)]

        head([[[torch.randn(4)], []], [[torch.randn(4)], pool]])

        ((context, context_mask),) = scoring_head.received_contexts
        assert context_mask.tolist() == [False, True]
        assert torch.allclose(context[1], torch.stack(pool).mean(dim=0))

    def test_returns_padded_logits_per_example(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(card_embedding_size=4)
        embeddings = [
            [[torch.randn(4), torch.randn(4)], []],  # 2 options, no pool
            [[torch.randn(4)], [torch.randn(4)]],  # 1 option, 1-card pool
        ]

        output = head(embeddings)

        assert output.shape == (2, 2)
        assert output[1, 1].item() == float("-inf")

    def test_an_empty_pool_scores_like_an_unconditioned_pick(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(card_embedding_size=4)
        options = [torch.randn(4), torch.randn(4)]

        with_empty_pool = head([[options, []], [[torch.randn(4)], [torch.randn(4)]]])
        unconditioned = head.scoring_head.score(
            torch.stack(options).unsqueeze(0),
            torch.ones(1, 2, dtype=torch.bool),
            None,
            None,
        )

        assert torch.allclose(with_empty_pool[0], unconditioned[0])

    def test_uses_injected_collaborators_by_default(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(card_embedding_size=4)

        assert isinstance(head.scoring_head, BilinearOptionScoringHead)
        assert isinstance(head.pooler, MeanEmbeddingPooler)
