"""Decoder head for the single-card-regression generic dojo cell.

Single card in (a SINGLE_CARD PaddedEmbeddings row), single scalar out per card - the
same architecture as v1's AveragePickNumberDecoderHead (see
src/dojos/seventeenlands/draft_game_metrics/average_pick_number/decoder_head.py),
now shared by every metric SingleCardRegressionDojo serves rather than
owned by one metric-specific dojo.
"""

import torch
import torch.nn as nn

from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape


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

    def forward(self, embeddings: PaddedEmbeddings) -> torch.Tensor:
        """One scalar prediction per input embedding, as a single (N,) tensor.

        Inputs:
            embeddings: a SINGLE_CARD batch, one card per example.
        Output: a (batch_size,) tensor of predictions, same order as
            embeddings.
        Side effects: none.
        Exceptions: none expected.
        """
        embeddings.require_shape(
            InputShape.SINGLE_CARD, "SingleCardRegressionDecoderHead"
        )
        # Each example's one card is at position 0, so no stack is needed;
        # run the network, then drop the trailing size-1 dim so callers
        # get a flat (batch_size,) tensor rather than (batch_size, 1).
        batched_embeddings = embeddings.cards[:, 0]
        decoder_output = self.network(batched_embeddings)
        return decoder_output.squeeze(-1)
