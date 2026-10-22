"""Cross-slice card match style (plans/multi_card_contrastive_dojos.md, A).

Per deck, two disjoint slices (DeckSlicesPairConstructor); each slice
attends within itself. A card's positives are the other slice's cards, its
own slice's cards are ignored, and other decks' cards are negatives:

    [a1 a2 a3]  [a4 a5 a6]  [b1 b2 b3]  [b4 b5 b6]
    a1: positives a4 a5 a6; ignored a2 a3; negatives b1 ... b6
"""

from dataclasses import dataclass
from uuid import UUID

import torch

from src.dojos.contrastive.contrastive_loss import (
    ComparisonMasks,
    ContrastiveLoss,
    check_item_embeddings,
    clique_positive_mask,
    constant_logit_baseline,
    info_nce_loss,
    info_nce_masks,
    same_key_mask,
)
from src.dojos.contrastive.pair_constructor import ContrastivePairConstructor
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.dojos.contrastive.styles.deck_slices import DeckSlicesPairConstructor
from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape


class CrossSliceCardInfoNCELoss:
    """Concrete ContrastiveLoss for the cross-slice card match style: each item is a
    group of cards (a slice from DeckSlicesPairConstructor), but the comparison
    unit is still the individual card, never a pooled whole-item vector.
    A card's positives are every card in a *different* item of its own
    item's positive clique (same source deck); cards in the anchor's own
    item are excluded entirely (neither positive nor negative), per the
    plan's explicit "an anchor is not a valid pair with other cards in
    its own set grouping" requirement - this is the one behavior
    genuinely new relative to SingleCardInfoNCELoss, expressed as the
    excluded mask passed to info_nce_masks."""

    def __init__(self, temperature: float = 0.07) -> None:
        """
        Inputs:
            temperature: InfoNCE's softmax temperature - see
                SingleCardInfoNCELoss.__init__.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if temperature <= 0.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature

    def calculate(
        self,
        item_embeddings: PaddedEmbeddings,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> torch.Tensor:
        """Mean per-card single-positive InfoNCE loss over the batch.

        Inputs: see ContrastiveLoss.calculate(). item_embeddings must be
            MULTI_CARD; identities[i][k] is the card at row i, position k.
        Output: a scalar loss tensor - the mean, over every card that
            has at least one cross-item positive, of that card's own
            InfoNCE loss.
        Side effects: none.
        Exceptions: ValueError if item_embeddings and identities disagree
            (count, shape or per-item length); DegenerateBatchError if no
            card in the batch has a positive pair (loss is undefined).

        Example:
            >>> loss_fn = CrossSliceCardInfoNCELoss()
            >>> loss_fn.calculate(item_embeddings, identities, positive_cliques)
        """
        # Validate inputs: per-item lengths must match, so flat card k
        # lines up with flat identity k
        check_item_embeddings(
            item_embeddings,
            identities,
            InputShape.MULTI_CARD,
            "CrossSliceCardInfoNCELoss",
        )
        _require_card_aligned_identities(item_embeddings, identities)
        # One unit per real card, in row-major order
        masks = self._masks(identities, positive_cliques)
        return info_nce_loss(item_embeddings.flat_cards(), masks, self._temperature)

    def constant_logit_loss(
        self,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> float:
        """See ContrastiveLoss.constant_logit_loss. The comparison unit is
        the flattened card; an anchor's own-item siblings are excluded, as
        in calculate().

        Example:
            >>> # 3 decks x 2 items x 5 cards: ln(5 * (6 - 2) + 1) = ln 21
            >>> CrossSliceCardInfoNCELoss().constant_logit_loss(
            ...     identities, [[0, 1], [2, 3], [4, 5]])
            3.044...
        """
        return constant_logit_baseline(self._masks(identities, positive_cliques))

    def _masks(
        self, identities: list[tuple[UUID, ...]], positive_cliques: list[list[int]]
    ) -> ComparisonMasks:
        """Card-level masks from item-level cliques.

        Private helper - callers are calculate() and constant_logit_loss().
        A card's positives are every card of a *different* item in its
        item's clique; its own item's other cards are excluded. Built by
        indexing the item-level clique mask with item_of_card (rows and
        columns), not by a per-card loop.
        Inputs: identities, positive_cliques - see calculate().
        Output: ComparisonMasks over the flat cards, in row-major order.
        Side effects: none. Exceptions: as clique_positive_mask.
        """
        flat_identities, item_of_card = _flat_card_identities(identities)
        item_positive = clique_positive_mask(positive_cliques, len(identities))
        positive = _expanded_to_cards(item_positive, item_of_card)
        own_item = same_key_mask(item_of_card)
        return info_nce_masks(positive, flat_identities, own_item)


def _require_card_aligned_identities(
    item_embeddings: PaddedEmbeddings, identities: list[tuple[UUID, ...]]
) -> None:
    """Each item's card count equals its identity's length, so flat card k
    is flat identity k.

    Private helper - single caller is CrossSliceCardInfoNCELoss.calculate.
    A deck-thinning mod that drops cards from an item keeps its identity,
    which would misalign the two; this loss cannot score such a batch.
    Inputs: item_embeddings, identities. Output: none.
    Side effects: none (host lengths). Exceptions: ValueError on any
        per-item mismatch.
    """
    identity_lengths = tuple(len(identity) for identity in identities)
    if item_embeddings.lengths != identity_lengths:
        raise ValueError(
            "CrossSliceCardInfoNCELoss needs one identity per card: item "
            f"lengths {item_embeddings.lengths} != identity lengths {identity_lengths}"
        )


def _flat_card_identities(
    identities: list[tuple[UUID, ...]],
) -> tuple[list[UUID], list[int]]:
    """Flatten per-item identity tuples to one uuid per card, tracking
    each card's item index.

    Private helper - single caller is CrossSliceCardInfoNCELoss._masks.
    Inputs: identities.
    Output: (flat_identities, item_of_card), both one entry per card, in
        item-then-position order (the row-major order of flat_cards()).
    Side effects: none. Exceptions: none.
    """
    flat_identities: list[UUID] = []
    item_of_card: list[int] = []
    for item_index, item_identities in enumerate(identities):
        flat_identities.extend(item_identities)
        item_of_card.extend([item_index] * len(item_identities))
    return flat_identities, item_of_card


def _expanded_to_cards(
    item_mask: torch.Tensor, item_of_card: list[int]
) -> torch.Tensor:
    """An item-level (items, items) mask read at card level.

    Private helper - single caller is CrossSliceCardInfoNCELoss._masks.
    Inputs: item_mask (items, items) bool CPU; item_of_card.
    Output: (cards, cards) bool CPU: item_mask[item_of_card][:, item_of_card].
    Side effects: none. Exceptions: none.
    """
    cards = torch.tensor(item_of_card, dtype=torch.long)
    return item_mask[cards][:, cards]


@dataclass(frozen=True)
class CrossSliceCardStyle:
    """The cross-slice card match style: DeckSlicesPairConstructor and
    CrossSliceCardInfoNCELoss.

    slice_size, slices_per_deck: see DeckSlicesPairConstructor. Two slices
        of 4 cost 8 cards per deck, in line with odd one out (8) and
        missing card (9), so the styles fit similar deck counts per batch; a deck
        costs slice_size x slices_per_deck cards of the batch budget.
    """

    slice_size: int = 4
    slices_per_deck: int = 2

    def pair_constructor(
        self, rng_seed: int, staple_subsampling: StapleSubsampling | None
    ) -> ContrastivePairConstructor:
        """See ContrastiveStyle.pair_constructor.

        Example:
            >>> CrossSliceCardStyle().pair_constructor(0, None).cards_per_deck
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
        return CrossSliceCardInfoNCELoss()
