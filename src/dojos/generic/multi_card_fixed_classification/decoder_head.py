"""Decoder head for the multi-card-fixed-classification generic dojo
cell.

Whole deck in (MultiCardEmbedding), one logit per candidate class out -
same pool-then-MLP shape as MultiCardRegressionDecoderHead (src/dojos/
generic/multi_card_regression/decoder_head.py, see src/dojos/generic/
pooling.py for the shared EmbeddingPooler Strategy) but the final layer
widens to num_classes instead of squeezing to one scalar, same as
FixedClassificationDecoderHead (src/dojos/generic/
single_card_fixed_classification/decoder_head.py) does over a single
card embedding instead of a pooled deck vector - so forward() stays a
rectangular (batch_size, num_classes) tensor, forward_single() a
(num_classes,) tensor, and there's no final squeeze(-1).
"""

from typing import List

import torch
import torch.nn as nn

from src.dojos.generic.pooling import EmbeddingPooler, MeanEmbeddingPooler
from src.schema.type_hints import BatchedMultiCardEmbedding, MultiCardEmbedding


class MultiCardFixedClassificationDecoderHead(nn.Module):
    """Pools a deck's card embeddings to one vector via a swappable
    EmbeddingPooler, then runs the same small MLP shape
    FixedClassificationDecoderHead uses to predict num_classes logits."""

    def __init__(
        self,
        card_embedding_size: int,
        num_classes: int,
        pooler: EmbeddingPooler | None = None,
    ) -> None:
        """
        Inputs:
            card_embedding_size: dimensionality of the card embeddings
                this head will receive (and of the pooled deck vector,
                since pooling doesn't change dimensionality).
            num_classes: len(label_values) of the FixedClassificationLoss
                this head's output will be scored against - fixes this
                head's output width.
            pooler: the EmbeddingPooler strategy this head delegates
                deck-collapsing to. None (default) uses
                MeanEmbeddingPooler.
        Output: none (constructor).
        Side effects: registers this module's layers (nn.Module state).
        Exceptions: none expected.
        """
        super().__init__()
        self.pooler = pooler or MeanEmbeddingPooler()
        # Same width/depth as MultiCardRegressionDecoderHead's MLP,
        # widened to num_classes on the output layer instead of 1 - same
        # substitution FixedClassificationDecoderHead makes over its
        # single-card regression sibling.
        self.network = nn.Sequential(
            nn.Linear(card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, embeddings: BatchedMultiCardEmbedding) -> torch.Tensor:
        """One num_classes-wide logits row per deck in the batch, as a
        single (batch_size, num_classes) tensor.

        Inputs:
            embeddings: one deck's card embeddings per training example
                in the batch (each deck's own list may be a different
                length).
        Output: a (batch_size, num_classes) tensor of logits, same row
            order as embeddings.
        Side effects: none.
        Exceptions: none expected.
        """
        # Pool each deck independently, stack into one batched tensor,
        # run the MLP. Unlike MultiCardRegressionDecoderHead.forward(),
        # no trailing-dim squeeze() at the end - num_classes is a real
        # dimension to keep, not a size-1 dim to drop.
        pooled_decks: List[torch.Tensor] = [
            self.pooler.pool(deck_embeddings) for deck_embeddings in embeddings
        ]
        batched_embeddings = torch.stack(pooled_decks)
        return self.network(batched_embeddings)

    def forward_single(self, embeddings: MultiCardEmbedding) -> torch.Tensor:
        """Decode one deck's card embeddings to its predicted class
        logits, unbatched.

        Inputs:
            embeddings: one deck's card embeddings.
        Output: a (num_classes,) tensor of logits.
        Side effects: none.
        Exceptions: none expected.
        """
        pooled_deck = self.pooler.pool(embeddings)
        return self.network(pooled_deck)
