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

from typing import List

import torch
import torch.nn as nn

from src.dojos.generic.option_scoring import (
    BilinearOptionScoringHead,
    OptionScoringHead,
)
from src.schema.type_hints import BatchedMultiCardEmbedding, MultiCardEmbedding


class MultiCardOptionSelectionDecoderHead(nn.Module):
    """Scores each example's ragged pack options independently via an
    injected OptionScoringHead, with no conditioning context."""

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

    def forward(self, embeddings: BatchedMultiCardEmbedding) -> List[torch.Tensor]:
        """One ragged logits tensor per example in the batch.

        Inputs:
            embeddings: one example's pack-option embeddings per
                training example in the batch - each example's own
                list may be a different length (pack size shrinks over
                the course of a draft).
        Output: a List[torch.Tensor], one 1-D logits tensor per
            example, same order as embeddings and same per-example
            length as that example's own option count - directly the
            shape PickPredictionCrossEntropyLoss.calculate() expects as
            decoder_output (no further stacking; the ragged lengths
            mean this can never collapse to one rectangular tensor).
        Side effects: none.
        Exceptions: none expected.

        Example:
            >>> head = MultiCardOptionSelectionDecoderHead(card_embedding_size=4)
            >>> len(head.forward([[torch.randn(4), torch.randn(4)]]))
            1
        """
        return [
            self.scoring_head.score(option_embeddings, context=None)
            for option_embeddings in embeddings
        ]

    def forward_single(self, embeddings: MultiCardEmbedding) -> torch.Tensor:
        """Decode one example's pack options to their logits, unbatched.

        Inputs:
            embeddings: one example's pack-option embeddings.
        Output: a (len(embeddings),) tensor of logits.
        Side effects: none.
        Exceptions: none expected.
        """
        return self.scoring_head.score(embeddings, context=None)
