"""Decoder head for the multi-card-regression generic dojo cell.

Whole deck in (MultiCardEmbedding - a variable-length list of card
embeddings, one deck's worth), single scalar out per deck. Unlike
SingleCardRegressionDecoderHead (src/dojos/generic/single_card_regression/
decoder_head.py), this cell's input isn't already one fixed-size
embedding - a deck is a variable-length list of them - so this head
first collapses each deck to one fixed-size vector via a privately
owned, swappable EmbeddingPooler (Strategy - PATTERNS.md, see
src/dojos/generic/pooling.py) before running the same small MLP shape
the single-card sibling uses. The pooling step is the one part of this
architecture expected to grow variants later (e.g. a learned attention
pooler); the MLP and the dojo's own constructor shape are not expected
to change when that happens - see plans/dojo_v2.md's contract manifest
for this cell.
"""

from typing import List

import torch
import torch.nn as nn

from src.dojos.generic.pooling import EmbeddingPooler, MeanEmbeddingPooler
from src.schema.type_hints import BatchedMultiCardEmbedding, MultiCardEmbedding


class MultiCardRegressionDecoderHead(nn.Module):
    """Pools a deck's card embeddings to one vector via a swappable
    EmbeddingPooler, then runs the same small MLP shape
    SingleCardRegressionDecoderHead uses to predict one scalar."""

    def __init__(
        self, card_embedding_size: int, pooler: EmbeddingPooler | None = None
    ) -> None:
        """
        Inputs:
            card_embedding_size: dimensionality of the card embeddings
                this head will receive (and of the pooled deck vector,
                since pooling doesn't change dimensionality).
            pooler: the EmbeddingPooler strategy this head delegates
                deck-collapsing to. None (default) uses
                MeanEmbeddingPooler.
        Output: none (constructor).
        Side effects: registers this module's layers (nn.Module state).
        Exceptions: none expected.
        """
        super().__init__()
        self.pooler = pooler or MeanEmbeddingPooler()
        # Same MLP shape as SingleCardRegressionDecoderHead - the pooled
        # deck vector has the same dimensionality as a single card
        # embedding, so no architecture change needed downstream of
        # pooling. Revisit width/depth once real training signal exists.
        self.network = nn.Sequential(
            nn.Linear(card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, embeddings: BatchedMultiCardEmbedding) -> torch.Tensor:
        """One scalar prediction per deck in the batch, as a single (N,) tensor.

        Inputs:
            embeddings: one deck's card embeddings per training example
                in the batch (each deck's own list may be a different
                length).
        Output: a (batch_size,) tensor of predictions, same order as
            embeddings.
        Side effects: none.
        Exceptions: none expected.
        """
        # Pool each deck independently (this is what actually handles
        # decks of varying card count - after this line every deck is
        # the same fixed-size vector), stack into one batched tensor,
        # run the MLP, then drop the trailing size-1 dim - same
        # stack-then-squeeze finish as SingleCardRegressionDecoderHead.
        pooled_decks: List[torch.Tensor] = [
            self.pooler.pool(deck_embeddings) for deck_embeddings in embeddings
        ]
        batched_embeddings = torch.stack(pooled_decks)
        decoder_output = self.network(batched_embeddings)
        return decoder_output.squeeze(-1)

    def forward_single(self, embeddings: MultiCardEmbedding) -> torch.Tensor:
        """Decode one deck's card embeddings to its predicted scalar, unbatched.

        Inputs:
            embeddings: one deck's card embeddings.
        Output: a scalar tensor.
        Side effects: none.
        Exceptions: none expected.
        """
        pooled_deck = self.pooler.pool(embeddings)
        return self.network(pooled_deck).squeeze(-1)
