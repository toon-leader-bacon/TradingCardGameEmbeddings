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
    ContrastiveLoss,
    check_item_embeddings,
    identity_negative_mask,
    mean_anchor_loss,
    mean_constant_logit_loss,
    pairwise_cosine_similarity,
)
from src.dojos.contrastive.pair_constructor import ContrastivePairConstructor
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.dojos.contrastive.styles.deck_slices import DeckSlicesPairConstructor
from src.schema.type_hints import (
    BatchedModelOutput,
    BatchedMultiCardEmbedding,
    Embedding,
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
    genuinely new relative to SingleCardInfoNCELoss, expressed via
    anchor_loss's excluded_indices argument."""

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

        flat_embeddings, flat_identities, item_of_card = self._flatten_items(
            multi_card_embeddings, identities
        )
        similarity = pairwise_cosine_similarity(flat_embeddings)
        valid_negative_mask = identity_negative_mask(flat_identities, similarity.device)
        positives_by_anchor, excluded_by_anchor = (
            self._positives_and_exclusions_by_anchor(item_of_card, positive_cliques)
        )

        # Average each anchor card's own InfoNCE loss - a card with no
        # cross-item positive in this batch contributes nothing.
        return mean_anchor_loss(
            positives_by_anchor,
            excluded_by_anchor,
            similarity,
            valid_negative_mask,
            self._temperature,
        )

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
        # Flatten identities to cards, then the same expansion calculate() uses
        flat_identities, item_of_card = self._flat_card_identities(identities)
        positives_by_anchor, excluded_by_anchor = (
            self._positives_and_exclusions_by_anchor(item_of_card, positive_cliques)
        )
        result = mean_constant_logit_loss(
            flat_identities, positives_by_anchor, excluded_by_anchor
        )
        return result

    def _flat_card_identities(
        self, identities: list[tuple[UUID, ...]]
    ) -> tuple[list[UUID], list[int]]:
        """Flatten per-item identity tuples to one uuid per card, tracking
        each card's item index - _flatten_items without the embeddings.

        Private helper - callers are constant_logit_loss() and
        _flatten_items().
        Inputs: identities - see calculate().
        Output: (flat_identities, item_of_card), both one entry per card,
            in item-then-position order.
        Side effects: none. Exceptions: none.
        """
        flat_identities: list[UUID] = []
        item_of_card: list[int] = []
        for item_index, item_identities in enumerate(identities):
            for card_identity in item_identities:
                flat_identities.append(card_identity)
                item_of_card.append(item_index)
        return flat_identities, item_of_card

    def _flatten_items(
        self,
        item_embeddings: BatchedMultiCardEmbedding,
        identities: list[tuple[UUID, ...]],
    ) -> tuple[list[Embedding], list[UUID], list[int]]:
        """Flatten every item's per-card embeddings/uuids into one flat
        card-level pool, tracking which item index each card came from.

        Private helper - single caller is calculate(). Builds its identity
        half with _flat_card_identities() (shared with
        constant_logit_loss()), adding only the embeddings.

        Inputs:
            item_embeddings: see calculate() - one MultiCardEmbedding
                per item.
            identities: see calculate() - identities[i] is positionally
                parallel to item_embeddings[i] (ContrastiveBatch's own
                contract), so identities[i][k] is the uuid of the card
                at item_embeddings[i][k].
        Output: (flat_embeddings, flat_identities, item_of_card), all
            the same length (total card count across every item):
            flat_embeddings[k] and flat_identities[k] are one card's
            embedding/uuid, and item_of_card[k] is the item index (into
            item_embeddings/identities) that card came from.
        Side effects: none.
        Exceptions: none expected (identities[i] is assumed the same
            length as item_embeddings[i] - a ContrastiveBatch/
            DeckSlicesPairConstructor invariant, not re-validated here).
        """
        flat_identities, item_of_card = self._flat_card_identities(identities)
        flat_embeddings = [embedding for item in item_embeddings for embedding in item]
        return flat_embeddings, flat_identities, item_of_card

    def _positives_and_exclusions_by_anchor(
        self, item_of_card: list[int], positive_cliques: list[list[int]]
    ) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
        """Expand item-level positive_cliques down to card-level
        positive/excluded index sets for every flattened card.

        Private helper - callers are calculate() and
        constant_logit_loss(). For a card at
        flat index k belonging to item i: its positives are every card
        belonging to a *different* item j in i's positive clique; its
        exclusions are every other card belonging to item i itself (see
        this class's own docstring).

        Inputs:
            item_of_card: item_of_card[k] is the item index flat card k
                came from - see _flatten_items().
            positive_cliques: see calculate().
        Output: (positives_by_anchor, excluded_by_anchor), both keyed by
            flat card index. A card with no cross-item positive in its
            clique is simply absent from positives_by_anchor (and
            anchor_loss is never called for it, so excluded_by_anchor
            need not have an entry for it either).
        Side effects: none.
        Exceptions: none.
        """
        cards_of_item: dict[int, list[int]] = {}
        for card_index, item_index in enumerate(item_of_card):
            cards_of_item.setdefault(item_index, []).append(card_index)
        positives_by_anchor: dict[int, list[int]] = {}
        excluded_by_anchor: dict[int, set[int]] = {}
        for clique in positive_cliques:
            for item_index in clique:
                own_cards = cards_of_item.get(item_index, [])
                other_cards = [
                    card
                    for other_item in clique
                    if other_item != item_index
                    for card in cards_of_item.get(other_item, [])
                ]
                if not other_cards:
                    continue
                for card in own_cards:
                    positives_by_anchor[card] = other_cards
                    excluded_by_anchor[card] = {
                        other for other in own_cards if other != card
                    }
        return positives_by_anchor, excluded_by_anchor


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
