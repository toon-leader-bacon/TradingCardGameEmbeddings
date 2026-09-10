"""Decoder head for the multi-card-binary-classification generic dojo
cell.

Whole deck in (MultiCardEmbedding - a variable-length list of card
embeddings, one deck's worth), single RAW LOGIT out per deck - NOT a
probability, no sigmoid applied here. BceLoss (src/dojos/loss/
bce_loss.py), the only consumer, applies nn.BCEWithLogitsLoss, which
expects raw logits for numerical stability (same reasoning
FixedClassificationLoss already relies on for its own un-softmaxed
per-class logits).

Structurally identical to MultiCardRegressionDecoderHead
(src/dojos/generic/multi_card_regression/decoder_head.py) - same pool-
stack-MLP-squeeze shape, same swappable EmbeddingPooler (Strategy -
PATTERNS.md, src/dojos/generic/pooling.py) - but kept as its own class
in its own file rather than collapsed into the regression sibling:
task shape (what the output number MEANS - a predicted value vs. a
pre-sigmoid logit) is a different concept from architectural shape,
even where the code looks the same today (PRINCIPLES.md section 2 -
"different concepts, prefer duplication over a coupled abstraction").
Same precedent as single_card_regression/decoder_head.py vs.
single_card_fixed_classification/decoder_head.py already being kept
separate.
"""

from typing import List

import torch
import torch.nn as nn

from src.dojos.generic.pooling import EmbeddingPooler, MeanEmbeddingPooler
from src.schema.type_hints import BatchedMultiCardEmbedding, MultiCardEmbedding


class MultiCardBinaryClassificationDecoderHead(nn.Module):
    """Pools a deck's card embeddings to one vector via a swappable
    EmbeddingPooler, then runs a small MLP to predict one raw logit."""

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
        # Same MLP shape as MultiCardRegressionDecoderHead - single
        # scalar out, here interpreted as a raw logit rather than a
        # predicted value. Revisit width/depth once real training
        # signal exists.
        self.network = nn.Sequential(
            nn.Linear(card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, embeddings: BatchedMultiCardEmbedding) -> torch.Tensor:
        """One raw logit per deck in the batch, as a single (N,) tensor.

        Inputs:
            embeddings: one deck's card embeddings per training example
                in the batch (each deck's own list may be a different
                length).
        Output: a (batch_size,) tensor of raw logits (no sigmoid
            applied), same order as embeddings.
        Side effects: none.
        Exceptions: none expected.
        """
        pooled_decks: List[torch.Tensor] = [
            self.pooler.pool(deck_embeddings) for deck_embeddings in embeddings
        ]
        batched_embeddings = torch.stack(pooled_decks)
        decoder_output = self.network(batched_embeddings)
        return decoder_output.squeeze(-1)

    def forward_single(self, embeddings: MultiCardEmbedding) -> torch.Tensor:
        """Decode one deck's card embeddings to its raw logit, unbatched.

        Inputs:
            embeddings: one deck's card embeddings.
        Output: a scalar tensor (a raw logit, no sigmoid applied).
        Side effects: none.
        Exceptions: none expected.
        """
        pooled_deck = self.pooler.pool(embeddings)
        return self.network(pooled_deck).squeeze(-1)
