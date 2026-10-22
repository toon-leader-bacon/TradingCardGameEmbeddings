"""Shared padding steps for the multi-group generic decoder heads.

Multi-group dojos still receive nested BatchedMultiGroupEmbedding (see
src/schema/padded_embeddings.py's ModelEmbeddings). Each multi-group head
reads a fixed group position (group 0 = the deck or the pack, group 1 =
the pool or the blockers) across every example, so it packs that one
group into PaddedEmbeddings and then uses the same batched pooler and
scorer as the multi-card heads. Shared by multi_group_binary_classification,
multi_group_regression and multi_group_option_selection.
"""

import torch

from src.dojos.generic.pooling import EmbeddingPooler
from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import BatchedMultiGroupEmbedding, InputShape


def padded_group(
    embeddings: BatchedMultiGroupEmbedding, group_index: int, width: int
) -> PaddedEmbeddings:
    """Group group_index of every example, as one MULTI_CARD batch.

    Inputs: embeddings (every example has more than group_index groups),
        group_index, width (d, for a batch where that group is empty
        everywhere).
    Output: PaddedEmbeddings with one row per example, in order; an empty
        group is an empty row.
    Side effects: none beyond autograd.
    Exceptions: IndexError if an example has too few groups.

    Example:
        >>> padded_group([[[a, b], []], [[c], [d]]], 1, 8).lengths
        (0, 1)
    """
    groups = [example[group_index] for example in embeddings]
    device = _device_of(embeddings)
    return PaddedEmbeddings.from_nested(groups, InputShape.MULTI_CARD, width, device)


def pooled_or_stand_in(
    pooled: torch.Tensor, group: PaddedEmbeddings, stand_in: torch.Tensor
) -> torch.Tensor:
    """pooled, with each empty example's row replaced by a learned vector.

    For heads whose second group may be empty ("no blockers"), which learn
    a stand-in for it rather than pooling nothing.
    Inputs: pooled (examples, d), the pool of group; group (the batch
        pooled came from); stand_in (d,), a learned parameter.
    Output: (examples, d); torch.where on has_cards() (built from host
        lengths, so no device sync), and the stand-in gets the gradient of
        exactly the rows it fills.
    Side effects: none beyond autograd. Exceptions: none.

    Example:
        >>> torch.equal(pooled_or_stand_in(pooled, group, stand_in)[1], stand_in)
        True  # when group.lengths == (2, 0)
    """
    has_cards = group.has_cards().unsqueeze(1)
    return torch.where(has_cards, pooled, stand_in.unsqueeze(0))


def pooled_group_pair(
    embeddings: BatchedMultiGroupEmbedding,
    pooler: EmbeddingPooler,
    empty_second_group: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Both groups of every example pooled and concatenated: the input of
    the symmetric two-group heads (binary classification, regression).

    Inputs: embeddings (each example [group_0, group_1]; group_0
        non-empty, group_1 may be empty), pooler, empty_second_group (the
        head's learned (d,) stand-in for an empty group_1), width (d).
    Output: (examples, 2 * width): [pool(group_0), pool(group_1) or the
        stand-in] per row.
    Side effects: none beyond autograd. Exceptions: none expected.

    Example:
        >>> pooled_group_pair(embeddings, MeanEmbeddingPooler(), stand_in, 4).shape
        torch.Size([2, 8])
    """
    # Pack and pool each group across the whole batch
    first = padded_group(embeddings, 0, width)
    second = padded_group(embeddings, 1, width)
    pooled_first = pooler.pool(first)
    pooled_second = pooled_or_stand_in(pooler.pool(second), second, empty_second_group)
    return torch.cat([pooled_first, pooled_second], dim=1)


def _device_of(embeddings: BatchedMultiGroupEmbedding) -> torch.device | None:
    """The device of the first card anywhere in embeddings, or None if
    there is no card (from_nested then builds on the CPU).

    Private helper - single caller is padded_group.
    Inputs: embeddings. Output: torch.device or None.
    Side effects: none. Exceptions: none.
    """
    for example in embeddings:
        for group in example:
            if group:
                return group[0].device
    return None
