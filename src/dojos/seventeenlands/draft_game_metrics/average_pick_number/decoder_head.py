"""Decoder head for the average-pick-number metric.

Single card in (SingleCardEmbedding), single scalar out per card: the
predicted average pick number. Rough sketch, pending discussion of
exactly what shape the embedding model hands us and how batching
should work here.
"""

import torch
import torch.nn as nn

from src.schema.type_hints import BatchedSingleCardEmbedding, SingleCardEmbedding


class AveragePickNumberDecoderHead(nn.Module):
    def __init__(self, card_embedding_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, embeddings: BatchedSingleCardEmbedding) -> torch.Tensor:
        """One scalar prediction per input embedding, as a single (N,) tensor."""
        # List[Embedding] -> (batch_size, embedding_dim)
        batched_embeddings = torch.stack(embeddings)
        # (batch_size, embedding_dim) -> (batch_size, 1)
        decoder_output = self.network(batched_embeddings)
        # Drop the trailing dim. Instead, we just want to return a flat tensor,
        # with one prediction per input embedding.
        return decoder_output.squeeze(-1)

    def forward_single(self, embedding: SingleCardEmbedding) -> torch.Tensor:
        """Demo/example: decode one card's embedding to its predicted
        average pick number, unbatched."""
        return self.network(embedding).squeeze(-1)
