import torch

from src.dojos.generic.multi_group_option_selection.decoder_head import (
    MultiGroupOptionSelectionDecoderHead,
)
from src.dojos.generic.multi_group_option_selection.dojo import _option_count_fn


class _RecordingScoringHead:
    def __init__(self) -> None:
        self.received: list = []

    def score(self, options, mask, context, context_mask):
        self.received.append((options, mask))
        return torch.zeros(mask.shape)


class TestSkipOptionHead:
    def test_no_skip_by_default(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(card_embedding_size=4)

        logits = head([[[torch.randn(4), torch.randn(4)], []]])

        assert head.skip_embedding is None
        assert logits.shape == (1, 2)

    def test_the_skip_sits_right_after_each_example_options(self) -> None:
        scoring_head = _RecordingScoringHead()
        head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size=4, scoring_head=scoring_head, learns_skip_option=True
        )
        long_pack = [torch.randn(4), torch.randn(4)]
        short_pack = [torch.randn(4)]

        logits = head([[long_pack, []], [short_pack, []]])

        assert logits.shape == (2, 3)
        options, mask = scoring_head.received[0]
        skip = head.skip_embedding
        assert skip is not None
        # Long pack: its two options, then the skip at index len(options)
        assert torch.equal(options[0, 0], long_pack[0])
        assert torch.equal(options[0, 2], skip)
        # Short pack: the skip at index 1, then padding
        assert torch.equal(options[1, 1], skip)
        assert mask.tolist() == [[True, True, True], [True, True, False]]

    def test_the_skip_embedding_is_trainable(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size=4, learns_skip_option=True
        )

        logits = head(
            [[[torch.randn(4)], [torch.randn(4)]], [[torch.randn(4)] * 2, []]]
        )
        logits[torch.isfinite(logits)].sum().backward()

        assert head.skip_embedding is not None
        assert head.skip_embedding.grad is not None
        assert any(p is head.skip_embedding for p in head.parameters())

    def test_the_option_list_the_caller_passed_is_not_extended(self) -> None:
        head = MultiGroupOptionSelectionDecoderHead(
            card_embedding_size=4, learns_skip_option=True
        )
        options = [torch.randn(4)]

        head([[options, []]])

        assert len(options) == 1


class TestOptionCountForTheBaseline:
    def test_counts_the_pack_alone_without_a_skip(self) -> None:
        count = _option_count_fn(can_skip=False)

        assert count([[object(), object()], []]) == 2  # type: ignore[list-item]

    def test_counts_the_skip_as_one_more_option(self) -> None:
        count = _option_count_fn(can_skip=True)

        assert count([[object(), object()], []]) == 3  # type: ignore[list-item]
