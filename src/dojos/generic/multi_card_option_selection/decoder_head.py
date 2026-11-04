"""Decoder head for the multi-card-option-selection generic dojo cell.

Ragged pack-option card embeddings in (MultiCardEmbedding - one group,
no conditioning context), one logit per option out - unlike every other
multi-card decoder head in this project (MultiCardRegressionDecoderHead
et al.), this never pools down to a single vector: this cell's whole
point is a per-option score, not one score for the whole set. Delegates
the actual per-option scoring to an injected OptionScoringHead Strategy
(src/dojos/generic/option_scoring.py) - see that module's docstring for
why scoring capacity is a swappable collaborator here rather than fixed
architecture. See multi_group_option_selection/decoder_head.py for the
sibling that also pools in a second (conditioning) card group.
"""

import torch
import torch.nn as nn

from src.dojos.generic.option_scoring import (
    BilinearOptionScoringHead,
    OptionScoringHead,
)
from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape


class MultiCardOptionSelectionDecoderHead(nn.Module):
    """Scores every example's pack options in one call to an injected
    OptionScoringHead, with no conditioning context."""

    def __init__(
        self, card_embedding_size: int, scoring_head: OptionScoringHead | None = None
    ) -> None:
        """
        Inputs:
            card_embedding_size: dimensionality of the card embeddings
                this head will receive.
            scoring_head: the OptionScoringHead strategy this head
                delegates per-option scoring to. None (default) uses
                BilinearOptionScoringHead.
        Output: none (constructor).
        Side effects: registers this module's layers (nn.Module state) -
            including scoring_head's own parameters, since it's an
            nn.Module (see OptionScoringHead's own docstring).
        Exceptions: none expected.
        """
        super().__init__()
        self.scoring_head = scoring_head or BilinearOptionScoringHead(
            card_embedding_size
        )

    def forward(self, embeddings: PaddedEmbeddings) -> torch.Tensor:
        """Every example's option logits, padded.

        Inputs:
            embeddings: a MULTI_CARD batch of pack options (pack size
                shrinks over the course of a draft, so lengths vary).
        Output: (examples, max_options) logits, -inf at padding: directly
            what PickPredictionCrossEntropyLoss.calculate() expects.
        Side effects: none.
        Exceptions: none expected.

        Example:
            >>> head = MultiCardOptionSelectionDecoderHead(card_embedding_size=4)
            >>> head.forward(padded_packs).shape
            torch.Size([2, 15])
        """
        embeddings.require_shape(
            InputShape.MULTI_CARD, "MultiCardOptionSelectionDecoderHead"
        )
        return self.scoring_head.score(
            embeddings.cards, embeddings.mask, context=None, context_mask=None
        )
