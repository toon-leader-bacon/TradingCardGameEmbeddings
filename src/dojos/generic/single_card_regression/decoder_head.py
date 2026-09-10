"""Decoder head for the single-card-regression generic dojo cell.

Single card in (SingleCardEmbedding), single scalar out per card - the
same architecture as v1's AveragePickNumberDecoderHead (see
src/dojos/seventeenlands/draft_game_metrics/average_pick_number/decoder_head.py),
now shared by every metric SingleCardRegressionDojo serves (see
plans/dojo_v2.md) rather than owned by one metric-specific dojo.
"""

import torch
import torch.nn as nn

from src.schema.type_hints import BatchedSingleCardEmbedding, SingleCardEmbedding


class SingleCardRegressionDecoderHead(nn.Module):
    """A small MLP mapping one card embedding to one scalar prediction."""

    def __init__(self, card_embedding_size: int) -> None:
        """
        Inputs:
            card_embedding_size: dimensionality of the card embeddings
                this head will receive.
        Output: none (constructor).
        Side effects: registers this module's layers (nn.Module state).
        Exceptions: none expected.
        """
        super().__init__()
        # Same shape as AveragePickNumberDecoderHead - a small MLP down
        # to one scalar. Revisit width/depth once real training signal
        # exists to tune against.
        self.network = nn.Sequential(
            nn.Linear(card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, embeddings: BatchedSingleCardEmbedding) -> torch.Tensor:
        """One scalar prediction per input embedding, as a single (N,) tensor.

        Inputs:
            embeddings: one embedding per training example in the batch.
        Output: a (batch_size,) tensor of predictions, same order as
            embeddings.
        Side effects: none.
        Exceptions: none expected.
        """
        # Stack the list of per-example embeddings into one batched
        # tensor, run it through the network, then drop the trailing
        # size-1 dim so callers get a flat (batch_size,) tensor rather
        # than (batch_size, 1).
        batched_embeddings = torch.stack(embeddings)
        decoder_output = self.network(batched_embeddings)
        return decoder_output.squeeze(-1)

    def forward_single(self, embedding: SingleCardEmbedding) -> torch.Tensor:
        """Decode one card's embedding to its predicted scalar, unbatched.

        Inputs:
            embedding: one card's embedding.
        Output: a scalar tensor.
        Side effects: none.
        Exceptions: none expected.
        """
        return self.network(embedding).squeeze(-1)
