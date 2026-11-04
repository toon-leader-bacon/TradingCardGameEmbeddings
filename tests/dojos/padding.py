"""Pack hand-built nested test embeddings into PaddedEmbeddings, the type
ContrastiveLoss.calculate() reads."""

from typing import Any

import torch

from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape


def padded(nested: Any) -> PaddedEmbeddings:
    """A list of tensors becomes a SINGLE_CARD batch, a list of tensor
    lists a MULTI_CARD one (the width is read from the first card)."""
    if isinstance(nested[0], torch.Tensor):
        return PaddedEmbeddings.from_nested(
            nested, InputShape.SINGLE_CARD, nested[0].shape[0]
        )
    width = next(card for item in nested for card in item).shape[0]
    return PaddedEmbeddings.from_nested(nested, InputShape.MULTI_CARD, width)
