"""Shared pooling strategy for multi-card generic dojo cells.

Collapses each deck's variable-length set of card embeddings into a
single fixed-size vector, for a whole padded batch at once. Originally written inside
multi_card_regression/decoder_head.py; moved here once a second
multi-card cell (multi_card_binary_classification) needed the exact
same pooling step ahead of a differently-shaped MLP/loss - genuinely
shared logic, not per-cell (PRINCIPLES.md section 3), so it lives
somewhere visible to every multi-card decoder head rather than being
duplicated or imported cross-cousin between cell directories.
"""

from typing import Protocol

import torch

from src.schema.padded_embeddings import PaddedEmbeddings


class EmbeddingPooler(Protocol):
    """Strategy: collapses each deck's variable-length set of card
    embeddings into a single fixed-size vector, for a whole padded batch
    at once. A multi-card decoder head owns one of these privately and
    delegates to it - swapping pooling behavior never requires touching a
    decoder head's own MLP or its owning dojo's constructor shape."""

    def pool(self, embeddings: PaddedEmbeddings) -> torch.Tensor:
        """Collapse each example's card embeddings into one vector.

        Inputs:
            embeddings: a MULTI_CARD batch (lengths vary per example).
        Output: an (examples, d) tensor, one row per example, same order.
            An empty example's row is implementation-defined; a head that
            must treat "empty" specially checks embeddings.lengths.
        Side effects: implementation-defined (expected: none for a
            pure pooling function; a future learned pooler may hold
            parameters but must still not mutate embeddings).
        Exceptions: implementation-defined (expected: none).
        """
        ...


class MeanEmbeddingPooler:
    """Default, only concrete EmbeddingPooler implemented so far -
    unweighted mean over a deck's card embeddings. Permutation-invariant
    (a deck is an unordered collection of cards) and has no learned
    parameters. A future AttentionPooler or similar is added as a new
    class in this same file and passed to a multi-card dojo's pooler=
    - no other file needs to change."""

    def pool(self, embeddings: PaddedEmbeddings) -> torch.Tensor:
        """Mean-pool every example's card embeddings.

        Inputs:
            embeddings: a MULTI_CARD batch.
        Output: (examples, d), embeddings.mean_per_example(); an empty
            example gives zeros.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> MeanEmbeddingPooler().pool(PaddedEmbeddings.from_nested(
            ...     [[torch.ones(4), torch.zeros(4)]], InputShape.MULTI_CARD, 4))
            tensor([[0.5000, 0.5000, 0.5000, 0.5000]])
        """
        return embeddings.mean_per_example()
