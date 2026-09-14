"""Decoder head for the multi-group-regression generic dojo cell.

Two card groups in (MultiGroupEmbedding = [group_0, group_1]), one
pooled-then-regressed scalar out - unlike multi_group_option_selection's
sibling decoder head (per-option classification logits), both groups
here contribute symmetrically to one aggregate label, so there's no
"which one" being scored: this pools each side to a single vector via a
shared EmbeddingPooler, concatenates the pair, and runs a small MLP
down to one scalar. See src/dojos/generic/multi_card_regression/
decoder_head.py for the MLP width/depth/squeeze precedent this doubles
the input width of, and multi_group_option_selection/decoder_head.py
for the empty-second-group handling this mirrors (though that cell
substitutes None/no-context; this one needs an actual fixed-size
vector to concatenate, hence the learned placeholder below).
"""

import torch
import torch.nn as nn

from src.dojos.generic.pooling import EmbeddingPooler, MeanEmbeddingPooler
from src.schema.type_hints import BatchedMultiGroupEmbedding, MultiGroupEmbedding


class MultiGroupRegressionDecoderHead(nn.Module):
    """Pools each of an example's two card groups (group 0, group 1) to
    one vector each via a shared EmbeddingPooler, concatenates them, and
    runs a small MLP down to one scalar prediction."""

    def __init__(
        self, card_embedding_size: int, pooler: EmbeddingPooler | None = None
    ) -> None:
        """
        Inputs:
            card_embedding_size: dimensionality of the card embeddings
                this head will receive (each pooled group vector shares
                this width; the concatenated pair is
                2 * card_embedding_size wide).
            pooler: the EmbeddingPooler strategy this head delegates
                BOTH groups' collapsing to - one shared instance, not
                one per group (mirrors every existing multi-card cell's
                "one pooler per decoder head" convention; a future
                learned pooler then ties both sides' pooling together,
                the right default for this symmetric two-sided
                concept). None (default) uses MeanEmbeddingPooler.
        Output: none (constructor).
        Side effects: registers this module's layers (nn.Module state),
            including a learned "empty second group" placeholder
            parameter (see forward_single()'s docstring).
        Exceptions: none expected.
        """
        super().__init__()
        self.pooler = pooler or MeanEmbeddingPooler()
        # A learned stand-in for group 1 (e.g. "no blockers") when it's
        # empty, rather than a plain zero vector - lets the model learn
        # what an absent second group should mean in the shared
        # embedding space instead of assuming the origin is neutral.
        # See forward_single()'s docstring.
        self.empty_second_group = nn.Parameter(torch.zeros(card_embedding_size))
        # Same width/depth as MultiCardRegressionDecoderHead's MLP,
        # doubled on the input side for the concatenated pair.
        self.network = nn.Sequential(
            nn.Linear(2 * card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, embeddings: BatchedMultiGroupEmbedding) -> torch.Tensor:
        """One scalar prediction per example, as a single (N,) tensor.

        Inputs:
            embeddings: one [group_0_embeddings, group_1_embeddings]
                pair per training example in the batch - group_1 may be
                empty for a given example (see class docstring).
        Output: a (batch_size,) tensor of predictions, same order as
            embeddings.
        Side effects: none.
        Exceptions: none expected.

        Example:
            >>> head = MultiGroupRegressionDecoderHead(card_embedding_size=4)
            >>> head([[[torch.randn(4)], []]]).shape
            torch.Size([1])
        """
        predictions = [self.forward_single(group) for group in embeddings]
        return torch.stack(predictions)

    def forward_single(self, embeddings: MultiGroupEmbedding) -> torch.Tensor:
        """Decode one example's [group_0, group_1] pair to its scalar
        prediction, unbatched.

        Inputs:
            embeddings: one example's [group_0_embeddings,
                group_1_embeddings] pair. group_1_embeddings may be
                empty (e.g. an unblocked attack has no blocker group) -
                self.empty_second_group stands in for it rather than
                calling self.pooler.pool([]) (undefined for an empty
                list - see MeanEmbeddingPooler's own docstring).
        Output: a scalar tensor.
        Side effects: none.
        Exceptions: none expected.
        """
        group_0_embeddings, group_1_embeddings = embeddings
        pooled_0 = self.pooler.pool(group_0_embeddings)
        pooled_1 = (
            self.pooler.pool(group_1_embeddings)
            if group_1_embeddings
            else self.empty_second_group
        )
        combined = torch.cat([pooled_0, pooled_1])
        return self.network(combined).squeeze(-1)
