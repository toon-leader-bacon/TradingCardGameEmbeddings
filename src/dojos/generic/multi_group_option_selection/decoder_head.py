"""Decoder head for the multi-group-option-selection generic dojo cell.

Pack-option card embeddings PLUS a second (pool) card group in
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
instead of always None. With learns_skip_option, one learned "pick none"
embedding is scored as a last extra option, so the logits run one longer.
"""

from typing import cast

import torch
import torch.nn as nn

from src.dojos.generic.option_scoring import (
    BilinearOptionScoringHead,
    OptionScoringHead,
)
from src.dojos.generic.pooling import EmbeddingPooler, MeanEmbeddingPooler
from src.dojos.generic.group_batching import padded_group
from src.schema.padded_embeddings import PaddedEmbeddings, mask_from_lengths
from src.schema.type_hints import BatchedMultiGroupEmbedding
from src.utils.device_copy import host_tensor

# Fixed group positions; see the module docstring
_OPTIONS_GROUP = 0
_POOL_GROUP = 1


class MultiGroupOptionSelectionDecoderHead(nn.Module):
    """Pools each example's pool group (if non-empty) to a context
    vector, then scores each of that example's pack options against it
    via an injected OptionScoringHead."""

    def __init__(
        self,
        card_embedding_size: int,
        scoring_head: OptionScoringHead | None = None,
        pooler: EmbeddingPooler | None = None,
        learns_skip_option: bool = False,
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
            learns_skip_option: True adds one more option after the real
                ones, "pick none of them", scored like a card by a
                learned embedding (skip_embedding). The logits then have
                len(options) + 1 entries and a skip's label is
                len(options). False (default): no such option.
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
        self._width = card_embedding_size
        # The learned stand-in for "pick none of them"; None when the
        # cell has no skip option
        self.skip_embedding: nn.Parameter | None = (
            nn.Parameter(torch.randn(card_embedding_size) * 0.02)
            if learns_skip_option
            else None
        )

    def forward(self, embeddings: BatchedMultiGroupEmbedding) -> torch.Tensor:
        """Every example's option logits, padded.

        Inputs:
            embeddings: one [option_embeddings, pool_embeddings] group
                pair per training example in the batch (see the module
                docstring for why the group order is fixed). A pool may be
                empty (a drafter's first pick has no pool yet).
        Output: (examples, max_options) logits, -inf at padding, where
            max_options counts the skip option when this head learns one.
            The skip sits right after each example's real options, before
            its padding, so a skip's label is still len(options).
        Side effects: none.
        Exceptions: none expected.

        Example:
            >>> head(embeddings).shape
            torch.Size([2, 16])
        """
        # Pack each group across the batch
        options = padded_group(embeddings, _OPTIONS_GROUP, self._width)
        pools = padded_group(embeddings, _POOL_GROUP, self._width)

        # The skip option scores like one more card, after the real ones
        option_cards, option_mask = options.cards, options.mask
        if self.skip_embedding is not None:
            option_cards, option_mask = self._with_skip_option(options)

        # An empty pool (first pick) is scored as unconditioned
        context = self.pooler.pool(pools)
        context_mask = pools.has_cards()
        return self.scoring_head.score(option_cards, option_mask, context, context_mask)

    def _with_skip_option(
        self, options: PaddedEmbeddings
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """options with skip_embedding written at each example's position
        lengths[b], one column wider.

        Private helper - single caller is forward.
        Inputs: options, a MULTI_CARD batch; skip_embedding is not None.
        Output: (cards (examples, max_options + 1, d), mask), still
            left-aligned. Positions come from host lengths; the write is
            out of place, so skip_embedding gets every row's gradient.
        Side effects: none beyond autograd; non-blocking host-to-device
            copies of the positions and mask (no device sync).
        Exceptions: none.
        """
        skip = cast(nn.Parameter, self.skip_embedding)
        examples, max_options, width = options.cards.shape

        # One column wider; the skip goes right after each example's real options
        positions = host_tensor(options.lengths, torch.long, options.cards.device)
        widened = torch.cat(
            [options.cards, options.cards.new_zeros((examples, 1, width))], dim=1
        )
        columns = torch.arange(max_options + 1, device=options.cards.device)
        at_skip = columns.unsqueeze(0) == positions.unsqueeze(1)

        # Out-of-place write, so skip_embedding gets every row's gradient
        cards = torch.where(at_skip.unsqueeze(2), skip.to(widened.dtype), widened)
        mask = mask_from_lengths(
            [length + 1 for length in options.lengths], cards.device
        )
        return cards, mask
