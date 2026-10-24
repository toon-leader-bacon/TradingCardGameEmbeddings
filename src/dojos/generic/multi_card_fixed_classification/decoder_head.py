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
rectangular (batch_size, num_classes) tensor, and there's no final
squeeze(-1).
"""

import torch
import torch.nn as nn

from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape

from src.dojos.generic.pooling import EmbeddingPooler, MeanEmbeddingPooler


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

    def forward(self, embeddings: PaddedEmbeddings) -> torch.Tensor:
        """One num_classes-wide logits row per deck in the batch, as a
        single (batch_size, num_classes) tensor.

        Inputs:
            embeddings: a MULTI_CARD batch, one deck per example.
        Output: a (batch_size, num_classes) tensor of logits, same row
            order as embeddings.
        Side effects: none.
        Exceptions: none expected.
        """
        embeddings.require_shape(
            InputShape.MULTI_CARD, "MultiCardFixedClassificationDecoderHead"
        )
        # Pool every deck at once (this is what handles decks of varying
        # card count: each row is now one fixed-size vector), then the MLP.
        # No trailing squeeze: num_classes is a real dimension to keep
        batched_embeddings = self.pooler.pool(embeddings)
        return self.network(batched_embeddings)
