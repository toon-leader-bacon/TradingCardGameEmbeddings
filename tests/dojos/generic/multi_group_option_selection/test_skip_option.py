import torch

from src.dojos.generic.multi_group_option_selection.decoder_head import (
    MultiGroupOptionSelectionDecoderHead,
)
from src.dojos.generic.multi_group_option_selection.dojo import _option_count_fn


class _RecordingScoringHead:
    def __init__(self) -> None:
        self.received_options: list = []

    def score(self, option_embeddings, context):
        self.received_options.append(list(option_embeddings))
        return torch.zeros(len(option_embeddings))


class TestSkipOptionHead:
    def test_no_skip_by_default(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(card_embedding_size=4)

        logits = head.forward_single([[torch.randn(4), torch.randn(4)], []])

        assert head.skip_embedding is None
        assert logits.shape == (2,)

    def test_a_skip_option_adds_one_logit_after_the_real_options(self) -> None:
        scoring_head = _RecordingScoringHead()
        head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size=4, scoring_head=scoring_head, learns_skip_option=True
        )
        options = [torch.randn(4), torch.randn(4)]

        logits = head.forward_single([options, []])

        assert logits.shape == (3,)
        scored = scoring_head.received_options[0]
        assert scored[-1] is head.skip_embedding
        assert scored[:2] == options

    def test_the_skip_embedding_is_trainable(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size=4, learns_skip_option=True
        )

        logits = head.forward_single([[torch.randn(4)], [torch.randn(4)]])
        logits.sum().backward()

        assert head.skip_embedding is not None
        assert head.skip_embedding.grad is not None
        assert any(p is head.skip_embedding for p in head.parameters())

    def test_the_option_list_the_caller_passed_is_not_extended(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size=4, learns_skip_option=True
        )
        options = [torch.randn(4)]

        head.forward_single([options, []])

        assert len(options) == 1


class TestOptionCountForTheBaseline:
    def test_counts_the_pack_alone_without_a_skip(self) -> None:
        count = _option_count_fn(can_skip=False)

        assert count([[object(), object()], []]) == 2  # type: ignore[list-item]

    def test_counts_the_skip_as_one_more_option(self) -> None:
        count = _option_count_fn(can_skip=True)

        assert count([[object(), object()], []]) == 3  # type: ignore[list-item]
