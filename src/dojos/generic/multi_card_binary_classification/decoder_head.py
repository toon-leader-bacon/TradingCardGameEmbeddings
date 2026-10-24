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

import torch
import torch.nn as nn

from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape

from src.dojos.generic.pooling import EmbeddingPooler, MeanEmbeddingPooler


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

    def forward(self, embeddings: PaddedEmbeddings) -> torch.Tensor:
        """One raw logit per deck in the batch, as a single (N,) tensor.

        Inputs:
            embeddings: a MULTI_CARD batch, one deck per example.
        Output: a (batch_size,) tensor of raw logits (no sigmoid
            applied), same order as embeddings.
        Side effects: none.
        Exceptions: none expected.
        """
        embeddings.require_shape(
            InputShape.MULTI_CARD, "MultiCardBinaryClassificationDecoderHead"
        )
        # Pool every deck at once (this is what handles decks of varying
        # card count: each row is now one fixed-size vector), then the MLP
        batched_embeddings = self.pooler.pool(embeddings)
        decoder_output = self.network(batched_embeddings)
        return decoder_output.squeeze(-1)
