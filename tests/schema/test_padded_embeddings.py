import pytest
import torch

from src.schema.padded_embeddings import (
    PaddedEmbeddings,
    flat_slot_index,
    mask_from_lengths,
    scatter_to_slots,
)
from src.schema.type_hints import InputShape

_MULTI = InputShape.MULTI_CARD
_SINGLE = InputShape.SINGLE_CARD


def _vectors(count: int, width: int = 3) -> list[torch.Tensor]:
    return [torch.full((width,), float(i + 1)) for i in range(count)]


def _ragged() -> tuple[PaddedEmbeddings, list[list[torch.Tensor]]]:
    """Examples of 2, 0 and 1 cards (an empty example in the middle)."""
    a, b, c = _vectors(3)
    nested = [[a, b], [], [c]]
    return PaddedEmbeddings.from_nested(nested, _MULTI, 3), nested


class TestMaskFromLengths:
    def test_is_left_aligned(self) -> None:
        mask = mask_from_lengths([2, 0, 1], torch.device("cpu"))
        assert mask.tolist() == [[True, True], [False, False], [True, False]]

    def test_no_lengths_is_an_empty_mask(self) -> None:
        assert mask_from_lengths([], torch.device("cpu")).shape == (0, 0)


class TestFromNested:
    def test_pads_to_the_longest_example_with_zeros(self) -> None:
        padded, _ = _ragged()

        assert padded.cards.shape == (3, 2, 3)
        assert padded.lengths == (2, 0, 1)
        assert torch.equal(padded.cards[1], torch.zeros(2, 3))
        assert torch.equal(padded.cards[2, 1], torch.zeros(3))

    def test_flat_cards_follow_input_order(self) -> None:
        padded, nested = _ragged()

        flat = [card for example in nested for card in example]
        assert torch.equal(padded.flat_cards(), torch.stack(flat))

    def test_round_trips_through_to_nested(self) -> None:
        padded, nested = _ragged()

        back = padded.to_nested()

        assert [len(example) for example in back] == [2, 0, 1]
        for got_example, want_example in zip(back, nested):
            for got, want in zip(got_example, want_example):
                assert torch.equal(got, want)

    def test_a_single_card_batch_round_trips(self) -> None:
        cards = _vectors(2)

        padded = PaddedEmbeddings.from_nested(cards, _SINGLE, 3)

        assert padded.cards.shape == (2, 1, 3)
        assert all(torch.equal(g, w) for g, w in zip(padded.to_nested(), cards))

    def test_an_all_empty_batch_uses_the_given_width(self) -> None:
        padded = PaddedEmbeddings.from_nested([[], []], _MULTI, 5)

        assert padded.cards.shape == (2, 0, 5)
        assert padded.width == 5

    def test_gradients_flow_back_to_each_card(self) -> None:
        cards = [torch.randn(3, requires_grad=True) for _ in range(3)]

        padded = PaddedEmbeddings.from_nested([cards[:2], cards[2:]], _MULTI, 3)
        padded.mean_per_example().sum().backward()

        assert all(card.grad is not None for card in cards)

    def test_rejects_a_card_of_the_wrong_width(self) -> None:
        with pytest.raises(ValueError, match="want"):
            PaddedEmbeddings.from_nested([[torch.zeros(4)]], _MULTI, 3)

    def test_rejects_a_multi_group_shape(self) -> None:
        with pytest.raises(ValueError):
            PaddedEmbeddings.from_nested([[torch.zeros(3)]], InputShape.MULTI_GROUP, 3)


class TestPerExampleReads:
    def test_mean_per_example_divides_by_the_real_length(self) -> None:
        padded, _ = _ragged()

        means = padded.mean_per_example()

        assert torch.allclose(means[0], torch.full((3,), 1.5))
        assert torch.equal(means[1], torch.zeros(3))  # empty example
        assert torch.allclose(means[2], torch.full((3,), 3.0))

    def test_example_of_card_maps_flat_cards_to_their_example(self) -> None:
        padded, _ = _ragged()
        assert padded.example_of_card().tolist() == [0, 0, 2]

    def test_has_cards_marks_non_empty_examples(self) -> None:
        padded, _ = _ragged()
        assert padded.has_cards().tolist() == [True, False, True]

    def test_len_is_the_example_count(self) -> None:
        padded, _ = _ragged()
        assert len(padded) == 3


class TestFromSingleCards:
    def test_each_row_is_a_one_card_example(self) -> None:
        units = torch.randn(4, 3)

        padded = PaddedEmbeddings.from_single_cards(units)

        assert padded.shape is _SINGLE
        assert padded.lengths == (1, 1, 1, 1)
        assert torch.equal(padded.cards[:, 0], units)

    def test_rejects_a_non_matrix(self) -> None:
        with pytest.raises(ValueError):
            PaddedEmbeddings.from_single_cards(torch.randn(4))


class TestValidation:
    def test_rejects_lengths_that_disagree_with_the_cards(self) -> None:
        cards = torch.zeros(2, 2, 3)
        mask = torch.ones(2, 2, dtype=torch.bool)
        with pytest.raises(ValueError, match="lengths"):
            PaddedEmbeddings(cards, mask, _MULTI, (2,))
        with pytest.raises(ValueError, match="range"):
            PaddedEmbeddings(cards, mask, _MULTI, (2, 3))

    def test_rejects_a_mismatched_mask(self) -> None:
        with pytest.raises(ValueError, match="mask"):
            PaddedEmbeddings(
                torch.zeros(2, 2, 3), torch.ones(2, 1, dtype=torch.bool), _MULTI, (1, 1)
            )

    def test_rejects_a_wide_single_card_batch(self) -> None:
        with pytest.raises(ValueError, match="SINGLE_CARD"):
            PaddedEmbeddings(
                torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), _SINGLE, (2,)
            )

    def test_rejects_an_empty_single_card_example(self) -> None:
        with pytest.raises(ValueError, match="exactly one card"):
            PaddedEmbeddings(
                torch.zeros(2, 1, 3),
                torch.tensor([[True], [False]]),
                _SINGLE,
                (1, 0),
            )

    def test_require_shape_names_the_reader(self) -> None:
        padded, _ = _ragged()
        padded.require_shape(_MULTI, "Head")
        with pytest.raises(ValueError, match="Head expects SINGLE_CARD"):
            padded.require_shape(_SINGLE, "Head")


class TestSlots:
    def test_flat_slot_index_is_row_major(self) -> None:
        assert flat_slot_index([2, 0, 1], 2, torch.device("cpu")).tolist() == [0, 1, 4]

    def test_scatter_to_slots_inverts_flat_cards(self) -> None:
        flat = torch.randn(3, 4, requires_grad=True)

        padded = scatter_to_slots(flat, torch.zeros(3, 2, 4), [2, 0, 1])
        padded.sum().backward()

        assert torch.equal(padded[0], flat[:2].detach())
        assert torch.equal(padded[2, 0], flat[2].detach())
        assert torch.equal(padded[1], torch.zeros(2, 4))
        assert flat.grad is not None and torch.equal(flat.grad, torch.ones(3, 4))
