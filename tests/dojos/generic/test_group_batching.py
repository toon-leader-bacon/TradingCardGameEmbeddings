import torch

from src.dojos.generic.group_batching import (
    padded_group,
    pooled_group_pair,
    pooled_or_stand_in,
)
from src.dojos.generic.pooling import MeanEmbeddingPooler


def _examples() -> list:
    """Two [group_0, group_1] examples; the first has an empty group 1."""
    return [
        [[torch.ones(2), torch.zeros(2)], []],
        [[torch.full((2,), 4.0)], [torch.full((2,), 2.0), torch.zeros(2)]],
    ]


def test_padded_group_packs_one_group_across_examples() -> None:
    second = padded_group(_examples(), 1, 2)

    assert second.lengths == (0, 2)
    assert second.cards.shape == (2, 2, 2)


def test_padded_group_of_all_empty_groups_has_the_given_width() -> None:
    examples = [[[torch.ones(3)], []], [[torch.ones(3)], []]]

    second = padded_group(examples, 1, 3)

    assert second.cards.shape == (2, 0, 3)


def test_pooled_or_stand_in_replaces_only_empty_rows() -> None:
    group = padded_group(_examples(), 1, 2)
    stand_in = torch.tensor([9.0, 9.0], requires_grad=True)

    result = pooled_or_stand_in(MeanEmbeddingPooler().pool(group), group, stand_in)
    result.sum().backward()

    assert torch.equal(result[0], stand_in.detach())
    assert torch.allclose(result[1], torch.tensor([1.0, 1.0]))
    assert stand_in.grad is not None
    assert stand_in.grad.tolist() == [1.0, 1.0]  # one row filled


def test_pooled_group_pair_concatenates_both_groups_in_order() -> None:
    stand_in = torch.tensor([9.0, 9.0])

    pairs = pooled_group_pair(_examples(), MeanEmbeddingPooler(), stand_in, 2)

    assert pairs.tolist() == [[0.5, 0.5, 9.0, 9.0], [4.0, 4.0, 1.0, 1.0]]
