"""Decoder head for the multi-group-binary-classification generic dojo
cell.

Two card groups in (MultiGroupEmbedding = [group_0, group_1]), one RAW
LOGIT out per example - no sigmoid here; BceLoss applies
nn.BCEWithLogitsLoss. Same pool-concatenate-MLP shape as
multi_group_regression/decoder_head.py (including the learned stand-in
for an empty group 1), kept as its own class for the same reason
multi_card_binary_classification keeps its head apart from the
multi_card_regression head: what the output number means (a logit, not
a value) is a different concept even where the code matches today.
"""

import torch
import torch.nn as nn

from src.dojos.generic.pooling import EmbeddingPooler, MeanEmbeddingPooler
from src.schema.type_hints import BatchedMultiGroupEmbedding, MultiGroupEmbedding


class MultiGroupBinaryClassificationDecoderHead(nn.Module):
    """Pools each of an example's two groups with one shared
    EmbeddingPooler (Strategy), concatenates [group_0, group_1] in that
    fixed order, and runs a small MLP down to one raw logit."""

    def __init__(
        self, card_embedding_size: int, pooler: EmbeddingPooler | None = None
    ) -> None:
        """
        Inputs:
            card_embedding_size: width of each card embedding (the
                concatenated pair is twice that).
            pooler: shared by both groups; None uses MeanEmbeddingPooler.
        Output: none (constructor).
        Side effects: registers the MLP and a learned empty-group-1
            placeholder parameter (nn.Module state).
        Exceptions: none.
        """
        super().__init__()
        self.pooler = pooler or MeanEmbeddingPooler()
        # Stands in for an empty group 1 (pooling an empty list is undefined)
        self.empty_second_group = nn.Parameter(torch.zeros(card_embedding_size))
        self.network = nn.Sequential(
            nn.Linear(2 * card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, embeddings: BatchedMultiGroupEmbedding) -> torch.Tensor:
        """One raw logit per example.

        Inputs: embeddings, one [group_0, group_1] embedding pair per
            example (group_1 may be empty).
        Output: a (batch_size,) tensor of raw logits, in input order.
        Side effects: none. Exceptions: none.

        Example:
            >>> head = MultiGroupBinaryClassificationDecoderHead(4)
            >>> head([[[torch.randn(4)], [torch.randn(4)]]]).shape
            torch.Size([1])
        """
        logits = [self.forward_single(group) for group in embeddings]
        return torch.stack(logits)

    def forward_single(self, embeddings: MultiGroupEmbedding) -> torch.Tensor:
        """One example's [group_0, group_1] pair -> its raw logit.

        Inputs: embeddings, one example's pair; group_1 may be empty
            (self.empty_second_group stands in for it).
        Output: a scalar tensor (raw logit).
        Side effects: none. Exceptions: none.
        """
        group_0_embeddings, group_1_embeddings = embeddings
        # Pool each side; an empty group 1 uses the learned placeholder
        pooled_0 = self.pooler.pool(group_0_embeddings)
        pooled_1 = (
            self.pooler.pool(group_1_embeddings)
            if group_1_embeddings
            else self.empty_second_group
        )
        # Fixed order: group 0 then group 1 (order carries meaning)
        return self.network(torch.cat([pooled_0, pooled_1])).squeeze(-1)
