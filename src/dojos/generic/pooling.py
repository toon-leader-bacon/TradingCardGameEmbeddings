"""Shared pooling strategy for multi-card generic dojo cells.

Collapses one deck's variable-length list of card embeddings into a
single fixed-size vector. Originally written inside
multi_card_regression/decoder_head.py; moved here once a second
multi-card cell (multi_card_binary_classification) needed the exact
same pooling step ahead of a differently-shaped MLP/loss - genuinely
shared logic, not per-cell (PRINCIPLES.md section 3), so it lives
somewhere visible to every multi-card decoder head rather than being
duplicated or imported cross-cousin between cell directories.
"""

from typing import Protocol

import torch

from src.schema.type_hints import MultiCardEmbedding


class EmbeddingPooler(Protocol):
    """Strategy: collapses one deck's variable-length list of card
    embeddings into a single fixed-size vector. A multi-card decoder
    head owns one of these privately and delegates to it - swapping
    pooling behavior never requires touching a decoder head's own MLP
    or its owning dojo's constructor shape."""

    def pool(self, embeddings: MultiCardEmbedding) -> torch.Tensor:
        """Collapse one deck's card embeddings into one vector.

        Inputs:
            embeddings: one deck's card embeddings, length equal to that
                deck's card count (varies deck to deck).
        Output: a single (card_embedding_size,) tensor representing the
            whole deck.
        Side effects: implementation-defined (expected: none for a pure
            pooling function; a future learned pooler may hold
            parameters but must still not mutate embeddings).
        Exceptions: implementation-defined (expected: none for a
            non-empty embeddings list - a decoder head's callers are
            responsible for never passing an empty deck).
        """
        ...


class MeanEmbeddingPooler:
    """Default, only concrete EmbeddingPooler implemented so far -
    unweighted mean over a deck's card embeddings. Permutation-invariant
    (a deck is an unordered collection of cards) and has no learned
    parameters. A future AttentionPooler or similar is added as a new
    class in this same file and passed to a multi-card dojo's pooler=
    - no other file needs to change."""

    def pool(self, embeddings: MultiCardEmbedding) -> torch.Tensor:
        """Mean-pool one deck's card embeddings.

        Inputs:
            embeddings: one deck's card embeddings (non-empty).
        Output: the elementwise mean, as a (card_embedding_size,)
            tensor.
        Side effects: none.
        Exceptions: none expected for a non-empty embeddings list (an
            empty list makes torch.stack raise - not this pooler's
            concern to guard against, see class docstring).

        Example:
            >>> MeanEmbeddingPooler().pool([torch.ones(4), torch.zeros(4)])
            tensor([0.5, 0.5, 0.5, 0.5])
        """
        return torch.stack(embeddings).mean(dim=0)
