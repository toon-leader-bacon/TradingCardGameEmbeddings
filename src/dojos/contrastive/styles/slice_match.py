"""Slice match style (plans/multi_card_contrastive_dojos.md, B).

The same batches as cross-slice card match (DeckSlicesPairConstructor),
but each slice is averaged into one vector after attention, and the
slices are compared like single cards:

    [a1 a2 a3] → A1    [a4 a5 a6] → A2    [b1 b2 b3] → B1    [b4 b5 b6] → B2
    A1: positive A2; negatives B1 B2

A deck-embedding objective: single cards matter only through the average.
"""

from dataclasses import dataclass
from typing import cast
from uuid import UUID

import torch

from src.dojos.contrastive.contrastive_loss import (
    ContrastiveLoss,
    check_item_embeddings,
    mean_item_embeddings,
)
from src.dojos.contrastive.pair_constructor import ContrastivePairConstructor
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.dojos.contrastive.styles.deck_slices import DeckSlicesPairConstructor
from src.dojos.contrastive.styles.single_card import SingleCardInfoNCELoss
from src.schema.type_hints import (
    BatchedModelOutput,
    BatchedMultiCardEmbedding,
    InputShape,
)


class PooledItemsLoss:
    """Decorator (PATTERNS.md) over a single-card ContrastiveLoss: averages
    each multi-card item's card embeddings into one vector, then hands the
    pooled items to the inner loss with the cliques unchanged. A pooled
    item is a multiset (equivariant attention, then a mean), so each
    identity is first made order-free (sorted): two slices of the same
    cards in another order embed identically and must not count as each
    other's negatives. calculate() and constant_logit_loss() both use
    those keys, so the baseline matches.
    """

    def __init__(self, inner: ContrastiveLoss) -> None:
        """
        Inputs: inner, a ContrastiveLoss over single-card-shaped items
            (e.g. SingleCardInfoNCELoss).
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._inner = inner

    def calculate(
        self,
        item_embeddings: BatchedModelOutput,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> torch.Tensor:
        """The inner loss over the items' mean embeddings.

        Inputs: see ContrastiveLoss.calculate(); item_embeddings must be
            multi-card-shaped, every item non-empty.
        Output: the inner loss's scalar tensor.
        Side effects: none.
        Exceptions: ValueError on a shape or length mismatch; whatever the
            inner loss raises.

        Example:
            >>> PooledItemsLoss(SingleCardInfoNCELoss()).calculate(
            ...     item_embeddings, identities, [[0, 1], [2, 3]])
        """
        # Validate inputs, then pool each item and delegate
        check_item_embeddings(
            item_embeddings, identities, InputShape.MULTI_CARD, "PooledItemsLoss"
        )
        pooled = mean_item_embeddings(cast(BatchedMultiCardEmbedding, item_embeddings))
        return self._inner.calculate(
            pooled, _order_free_identities(identities), positive_cliques
        )

    def constant_logit_loss(
        self,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> float:
        """See ContrastiveLoss.constant_logit_loss: the inner loss's, on
        the order-free identities calculate() scores.

        Example:
            >>> PooledItemsLoss(SingleCardInfoNCELoss()).constant_logit_loss(
            ...     identities, [[0, 1], [2, 3]])
            1.098...
        """
        return self._inner.constant_logit_loss(
            _order_free_identities(identities), positive_cliques
        )


def _order_free_identities(
    identities: list[tuple[UUID, ...]],
) -> list[tuple[UUID, ...]]:
    """Each identity sorted, so equal multisets compare equal.

    Private helper - callers are PooledItemsLoss.calculate() and
    constant_logit_loss().
    Inputs: identities. Output: list of sorted tuples, same order.
    Side effects: none. Exceptions: none.
    """
    return [tuple(sorted(identity)) for identity in identities]


@dataclass(frozen=True)
class SliceMatchStyle:
    """The slice match style: DeckSlicesPairConstructor and
    PooledItemsLoss(SingleCardInfoNCELoss()).

    slice_size, slices_per_deck: see DeckSlicesPairConstructor. Two slices
        of 4 cost 8 cards per deck, in line with odd one out (8) and
        missing card (9), so the styles fit similar deck counts per batch.
    """

    slice_size: int = 4
    slices_per_deck: int = 2

    def pair_constructor(
        self, rng_seed: int, staple_subsampling: StapleSubsampling | None
    ) -> ContrastivePairConstructor:
        """See ContrastiveStyle.pair_constructor.

        Example:
            >>> SliceMatchStyle().pair_constructor(0, None).cards_per_deck
            8
        """
        return DeckSlicesPairConstructor(
            self.slice_size,
            self.slices_per_deck,
            rng_seed=rng_seed,
            staple_subsampling=staple_subsampling,
        )

    def contrastive_loss(self) -> ContrastiveLoss:
        """See ContrastiveStyle.contrastive_loss."""
        return PooledItemsLoss(SingleCardInfoNCELoss())
