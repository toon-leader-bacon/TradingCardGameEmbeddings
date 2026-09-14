import torch

from src.dojos.generic.multi_card_option_selection.decoder_head import (
    MultiCardOptionSelectionDecoderHead,
)
from src.dojos.generic.option_scoring import BilinearOptionScoringHead


class _RecordingScoringHead:
    """Records the context each call was given, returns a fixed-length
    zero vector - isolates the decoder head's own per-example iteration
    from BilinearOptionScoringHead's actual math."""

    def __init__(self) -> None:
        self.received_contexts: list = []

    def score(self, option_embeddings, context):
        self.received_contexts.append(context)
        return torch.zeros(len(option_embeddings))


class TestForward:
    def test_returns_one_ragged_logits_tensor_per_example(self) -> None:
        head = MultiCardOptionSelectionDecoderHead(card_embedding_size=4)
        embeddings = [
            [torch.randn(4), torch.randn(4)],  # a 2-option pack
            [torch.randn(4)],  # a 1-option pack - packs vary in size
        ]

        output = head(embeddings)

        assert [logits.shape[0] for logits in output] == [2, 1]

    def test_uses_injected_scoring_head_by_default_bilinear(self) -> None:
        default_head = MultiCardOptionSelectionDecoderHead(card_embedding_size=4)
        stub_head = MultiCardOptionSelectionDecoderHead(
            card_embedding_size=4, scoring_head=_RecordingScoringHead()
        )
        assert isinstance(default_head.scoring_head, BilinearOptionScoringHead)
        assert isinstance(stub_head.scoring_head, _RecordingScoringHead)

    def test_never_passes_a_context_to_the_scoring_head(self) -> None:
        scoring_head = _RecordingScoringHead()
        head = MultiCardOptionSelectionDecoderHead(
            card_embedding_size=4, scoring_head=scoring_head
        )
        embeddings = [[torch.randn(4)], [torch.randn(4), torch.randn(4)]]

        head(embeddings)

        assert scoring_head.received_contexts == [None, None]


class TestForwardSingle:
    def test_returns_one_logit_per_option(self) -> None:
        head = MultiCardOptionSelectionDecoderHead(card_embedding_size=4)

        result = head.forward_single([torch.randn(4), torch.randn(4)])

        assert result.shape == (2,)
