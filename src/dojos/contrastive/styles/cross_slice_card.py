"""Cross-slice card match style (plans/multi_card_contrastive_dojos.md, A).

Per deck, two disjoint slices (DeckSlicesPairConstructor); each slice
attends within itself. A card's positives are the other slice's cards, its
own slice's cards are ignored, and other decks' cards are negatives:

    [a1 a2 a3]  [a4 a5 a6]  [b1 b2 b3]  [b4 b5 b6]
    a1: positives a4 a5 a6; ignored a2 a3; negatives b1 ... b6
"""

from dataclasses import dataclass
from typing import cast
from uuid import UUID

import torch

from src.dojos.contrastive.contrastive_loss import (
    AnchorMasks,
    ContrastiveLoss,
    check_item_embeddings,
    clique_keys,
    constant_logit_baseline,
    masked_info_nce_loss,
    pairwise_cosine_similarity,
    same_key_mask,
    unit_keys,
)
from src.dojos.contrastive.pair_constructor import ContrastivePairConstructor
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.dojos.contrastive.styles.deck_slices import DeckSlicesPairConstructor
from src.schema.type_hints import (
    BatchedModelOutput,
    BatchedMultiCardEmbedding,
    InputShape,
)


class CrossSliceCardInfoNCELoss:
    """Concrete ContrastiveLoss for the cross-slice card match style: each item is a
    group of cards (a slice from DeckSlicesPairConstructor), but the comparison
    unit is still the individual card, never a pooled whole-item vector.
    A card's positives are every card in a *different* item of its own
    item's positive clique (same source deck); cards in the anchor's own
    item are excluded entirely (neither positive nor negative), per the
    plan's explicit "an anchor is not a valid pair with other cards in
    its own set grouping" requirement - this is the one behavior
    genuinely new relative to SingleCardInfoNCELoss, expressed as
    AnchorMasks.of's ignored mask."""

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
        item_embeddings: BatchedModelOutput,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> torch.Tensor:
        """Mean per-card single-positive InfoNCE loss over the batch.

        Inputs: see ContrastiveLoss.calculate(). item_embeddings must be
            multi-card-shaped (one MultiCardEmbedding - a list of
            per-card Embedding - per item).
        Output: a scalar loss tensor - the mean, over every card that
            has at least one cross-item positive, of that card's own
            InfoNCE loss.
        Side effects: none.
        Exceptions: ValueError if len(item_embeddings) != len(identities),
            if item_embeddings isn't multi-card-shaped, or if no card in
            the batch has a positive pair (loss is undefined).

        Example:
            >>> loss_fn = CrossSliceCardInfoNCELoss()
            >>> loss_fn.calculate(item_embeddings, identities, positive_cliques)
        """
        check_item_embeddings(
            item_embeddings,
            identities,
            InputShape.MULTI_CARD,
            "CrossSliceCardInfoNCELoss",
        )
        # Runtime shape already validated above - narrow the wide
        # ContrastiveLoss.calculate() parameter type back to the
        # concrete shape this class actually handles.
        multi_card_embeddings = cast(BatchedMultiCardEmbedding, item_embeddings)

        # One flat pool of cards, item by item, in identity order
        flat_embeddings = [
            embedding for item in multi_card_embeddings for embedding in item
        ]
        similarity = pairwise_cosine_similarity(flat_embeddings)
        masks = self._anchor_masks(identities, positive_cliques, similarity.device)

        # Average each anchor card's own InfoNCE loss - a card with no
        # cross-item positive in this batch contributes nothing.
        return masked_info_nce_loss(similarity, masks, self._temperature)

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
        result: float
        # The same masks calculate() uses, built on the CPU
        masks = self._anchor_masks(identities, positive_cliques, torch.device("cpu"))
        result = constant_logit_baseline(masks)
        return result

    def _flat_card_identities(
        self, identities: list[tuple[UUID, ...]]
    ) -> tuple[list[UUID], list[int]]:
        """Flatten per-item identity tuples to one uuid per card, tracking
        each card's item index.

        Private helper - single caller is _anchor_masks().
        Inputs: identities - see calculate().
        Output: (flat_identities, item_of_card), both one entry per card,
            in item-then-position order (the order calculate() flattens
            the embeddings in).
        Side effects: none. Exceptions: none.
        """
        flat_identities: list[UUID] = []
        item_of_card: list[int] = []
        for item_index, item_identities in enumerate(identities):
            for card_identity in item_identities:
                flat_identities.append(card_identity)
                item_of_card.append(item_index)
        return flat_identities, item_of_card

    def _anchor_masks(
        self,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
        device: torch.device,
    ) -> AnchorMasks:
        """Every flattened card's positives and ignored cards. For a card
        of item i: its positives are the cards of the other items in i's
        clique; the other cards of i itself are ignored (see this class's
        docstring); every other card is a negative unless it is the same
        card (exact-identity duplicate).

            positives = same(clique of item) and not same(item)
            ignored   = same(item)

        The clique keys are built per item, then gathered to cards
        through item_of_card.

        Private helper - callers are calculate() and
        constant_logit_loss().
        Inputs: identities, positive_cliques (see calculate()), device
            (where the masks go).
        Output: AnchorMasks over the flattened cards, on device.
        Side effects: one host sync (AnchorMasks' checks).
        Exceptions: ValueError if a clique index is out of range or an
            item is in two cliques.
        """
        flat_identities, item_of_card = self._flat_card_identities(identities)
        item_keys = torch.tensor(item_of_card, dtype=torch.int64, device=device)

        # Each card takes its item's clique key
        item_clique_keys = clique_keys(positive_cliques, len(identities), device)
        same_clique = same_key_mask(item_clique_keys[item_keys])
        same_item = same_key_mask(item_keys)
        return AnchorMasks.of(
            same_clique & ~same_item, same_item, unit_keys(flat_identities, device)
        )


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
