"""Decoder head for the multi-group-option-selection generic dojo cell.

Ragged pack-option card embeddings PLUS a second (pool) card group in
(MultiGroupEmbedding = [option_embeddings, pool_embeddings] - **group 0
is always the pack options, group 1 is always the pool** - see
src/dojos/generic/data_constructors.py's PoolConditionedPickDataConstructor
docstring for why this order is load-bearing, not arbitrary), one logit
per option out. Pools the pool group to a single context vector via a
swappable EmbeddingPooler (src/dojos/generic/pooling.py) before handing
each option embedding, plus that context, to an injected
OptionScoringHead (src/dojos/generic/option_scoring.py) - the same
strategy class multi_card_option_selection's decoder head uses, just
always given a real (possibly None, if the pool was empty) context here
instead of always None.
"""

from typing import List

import torch
import torch.nn as nn

from src.dojos.generic.option_scoring import (
    BilinearOptionScoringHead,
    OptionScoringHead,
)
from src.dojos.generic.pooling import EmbeddingPooler, MeanEmbeddingPooler
from src.schema.type_hints import BatchedMultiGroupEmbedding, MultiGroupEmbedding


class MultiGroupOptionSelectionDecoderHead(nn.Module):
    """Pools each example's pool group (if non-empty) to a context
    vector, then scores each of that example's pack options against it
    via an injected OptionScoringHead."""

    def __init__(
        self,
        card_embedding_size: int,
        scoring_head: OptionScoringHead | None = None,
        pooler: EmbeddingPooler | None = None,
    ) -> None:
        """
        Inputs:
            card_embedding_size: dimensionality of the card embeddings
                this head will receive.
            scoring_head: the OptionScoringHead strategy this head
                delegates per-option scoring to. None (default) uses
                BilinearOptionScoringHead - see
                multi_card_option_selection/decoder_head.py's sibling
                for the same default.
            pooler: the EmbeddingPooler strategy this head delegates
                pool-collapsing to. None (default) uses
                MeanEmbeddingPooler.
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
        self.pooler = pooler or MeanEmbeddingPooler()

    def forward(self, embeddings: BatchedMultiGroupEmbedding) -> List[torch.Tensor]:
        """One ragged logits tensor per example in the batch.

        Inputs:
            embeddings: one [option_embeddings, pool_embeddings] group
                pair per training example in the batch (see class
                docstring for why the group order is fixed).
        Output: a List[torch.Tensor], one 1-D logits tensor per
            example - see MultiCardOptionSelectionDecoderHead.forward()
            for why this needs no further stacking.
        Side effects: none.
        Exceptions: none expected.
        """
        return [self.forward_single(group) for group in embeddings]

    def forward_single(self, embeddings: MultiGroupEmbedding) -> torch.Tensor:
        """Decode one example's [option_embeddings, pool_embeddings] to
        its option logits, unbatched.

        Inputs:
            embeddings: one example's [option_embeddings,
                pool_embeddings] group pair. pool_embeddings may be
                empty (a drafter's first pick of a draft has no pool
                yet) - see the empty-pool guard below.
        Output: a (len(option_embeddings),) tensor of logits.
        Side effects: none.
        Exceptions: none expected.
        """
        option_embeddings, pool_embeddings = embeddings
        # An empty pool (first pick of a draft) has nothing to pool -
        # score with no conditioning context at all, same as
        # multi_card_option_selection's unconditioned case, rather than
        # calling self.pooler.pool([]) (undefined for an empty list -
        # see MeanEmbeddingPooler's own docstring).
        context = self.pooler.pool(pool_embeddings) if pool_embeddings else None
        return self.scoring_head.score(option_embeddings, context=context)
