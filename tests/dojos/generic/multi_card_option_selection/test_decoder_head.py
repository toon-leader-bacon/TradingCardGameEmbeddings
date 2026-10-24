import pytest
import torch

from src.dojos.generic.multi_card_option_selection.decoder_head import (
    MultiCardOptionSelectionDecoderHead,
)
from src.dojos.generic.option_scoring import BilinearOptionScoringHead
from tests.dojos.padding import padded


class _RecordingScoringHead:
    """Records each call's context, returns zero logits - isolates the
    decoder head's own wiring from BilinearOptionScoringHead's math."""

    def __init__(self) -> None:
        self.received_contexts: list = []

    def score(self, options, mask, context, context_mask):
        self.received_contexts.append((context, context_mask))
        return torch.zeros(mask.shape)


class TestForward:
    def test_returns_padded_logits_with_minus_inf_at_padding(self) -> None:
        head = MultiCardOptionSelectionDecoderHead(card_embedding_size=4)
        embeddings = padded(
            [
                [torch.randn(4), torch.randn(4)],  # a 2-option pack
                [torch.randn(4)],  # a 1-option pack - packs vary in size
            ]
        )

        output = head(embeddings)

        assert output.shape == (2, 2)
        assert output[1, 1].item() == float("-inf")
        assert bool(torch.isfinite(output[embeddings.mask]).all())

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

        head(padded([[torch.randn(4)], [torch.randn(4), torch.randn(4)]]))

        assert scoring_head.received_contexts == [(None, None)]

    def test_rejects_a_single_card_batch(self) -> None:
        head = MultiCardOptionSelectionDecoderHead(card_embedding_size=4)
        with pytest.raises(ValueError, match="MULTI_CARD"):
            head(padded([torch.randn(4)]))
