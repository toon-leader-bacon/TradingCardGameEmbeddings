"""Decoder head for the single-card-fixed-classification generic dojo
cell.

Single card in (SingleCardEmbedding), one logit per candidate class out
- same small-MLP shape as SingleCardRegressionDecoderHead (see
src/dojos/generic/single_card_regression/decoder_head.py) but the final
layer widens to num_classes instead of squeezing to one scalar, and
there's no final squeeze(-1): forward() stays a rectangular
(batch_size, num_classes) tensor, forward_single() a (num_classes,)
tensor.
"""

import torch
import torch.nn as nn

from src.schema.type_hints import BatchedSingleCardEmbedding, SingleCardEmbedding


class FixedClassificationDecoderHead(nn.Module):
    """A small MLP mapping one card embedding to num_classes logits."""

    def __init__(self, card_embedding_size: int, num_classes: int) -> None:
        """
        Inputs:
            card_embedding_size: dimensionality of the card embeddings
                this head will receive.
            num_classes: len(label_values) of the FixedClassificationLoss
                this head's output will be scored against - fixes this
                head's output width.
        Output: none (constructor).
        Side effects: registers this module's layers (nn.Module state).
        Exceptions: none expected.
        """
        super().__init__()
        # Same width/depth as SingleCardRegressionDecoderHead's MLP,
        # widened to num_classes on the output layer instead of 1.
        self.network = nn.Sequential(
            nn.Linear(card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, embeddings: BatchedSingleCardEmbedding) -> torch.Tensor:
        """One num_classes-wide logits row per input embedding, as a
        single (batch_size, num_classes) tensor.

        Inputs:
            embeddings: one embedding per training example in the batch.
        Output: a (batch_size, num_classes) tensor of logits, same row
            order as embeddings.
        Side effects: none.
        Exceptions: none expected.
        """
        # Stack the list of per-example embeddings into one batched
        # tensor, run it through self.network. Unlike
        # SingleCardRegressionDecoderHead.forward(), no trailing-dim
        # squeeze() at the end - num_classes is a real dimension to
        # keep, not a size-1 dim to drop.
        batched_embeddings = torch.stack(embeddings)
        return self.network(batched_embeddings)

    def forward_single(self, embedding: SingleCardEmbedding) -> torch.Tensor:
        """Decode one card's embedding to its predicted class logits,
        unbatched.

        Inputs:
            embedding: one card's embedding.
        Output: a (num_classes,) tensor of logits.
        Side effects: none.
        Exceptions: none expected.
        """
        return self.network(embedding)
