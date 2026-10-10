"""Single-card style: single cards, every same-deck card a positive."""

import logging
import random
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import torch

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.contrastive_loss import (
    ContrastiveLoss,
    check_item_embeddings,
    mean_anchor_loss,
    mean_constant_logit_loss,
    pairwise_cosine_similarity,
    identity_negative_mask,
)
from src.dojos.contrastive.pair_constructor import (
    ContrastivePairConstructor,
    SliceSampler,
    card_for_known_uuid,
)
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.schema.card import GenericCard, GenericDeck
from src.schema.type_hints import (
    BatchedModelOutput,
    BatchedSingleCardEmbedding,
    InputShape,
)

_logger = logging.getLogger(__name__)


class SingleCardPairConstructor:
    """Concrete ContrastivePairConstructor for this slice: single-card
    items only, drawn evenly across the given decks. A deck with fewer
    known cards than items_per_deck is skipped outright (its
    nocab_uuid logged) rather than partially sampled - a sampling
    *count* per deck, never full combinatorial enumeration of a deck's
    possible subsets (see plan)."""

    def __init__(
        self,
        items_per_deck: int,
        rng_seed: int | None = None,
        staple_subsampling: StapleSubsampling | None = None,
    ) -> None:
        """
        Inputs:
            items_per_deck: how many single-card items to sample from
                each deck that has at least this many known cards.
            rng_seed: seed for item sampling. None means
                non-deterministic.
            staple_subsampling: when given, each deck's known cards are
                first thinned (each occurrence kept with probability
                min(1, sqrt(t / df)), see staple_subsampling.py), and a
                deck left with fewer than items_per_deck cards is skipped
                like a deck with too few known cards: falling back to the
                unthinned deck would put staple pairs back exactly where
                staples dominate. None (the default, t = inf) samples
                as before and draws nothing extra from the RNG.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if items_per_deck <= 0.
        """
        if items_per_deck <= 0:
            raise ValueError("items_per_deck must be positive")
        self._items_per_deck = items_per_deck
        self._sampler = SliceSampler(random.Random(rng_seed), staple_subsampling)

    @property
    def cards_per_deck(self) -> int:
        """One card per item."""
        return self._items_per_deck

    def build(
        self, decks: list[GenericDeck], card_lookup: CardLookup
    ) -> ContrastiveBatch:
        """Sample items_per_deck single-card items from each deck, and
        mark every same-deck item pair as positive.

        Inputs: see ContrastivePairConstructor.build().
        Output: a ContrastiveBatch of single-card items. If every given
            deck is skipped (too few known cards, or too few left after
            staple subsampling), items/identities/positive_cliques are
            all empty.
        Side effects: emits one logging.warning() per deck skipped for
            too few known cards, and one logging.debug() per deck skipped
            only after staple subsampling (common for a tiny t).
        Exceptions: RuntimeError if card_lookup stops resolving a uuid
            it had just resolved moments earlier in this same call (an
            environment invariant violation, not expected/dirty-data
            territory). Also whatever ContrastiveBatch.__post_init__()
            raises, if this method's own bookkeeping produces an
            inconsistent batch (a bug, not an expected runtime
            condition).

        Example:
            >>> constructor = SingleCardPairConstructor(items_per_deck=11)
            >>> batch = constructor.build(deck_sample, card_lookup)
        """
        items: list[GenericCard] = []
        identities: list[tuple[UUID, ...]] = []
        positive_cliques: list[list[int]] = []

        # Sample each deck independently, tracking this deck's index
        # range in the flat `items` pool - that whole range is one
        # positive clique, since every item drawn from the same deck is
        # mutually positive.
        for deck in decks:
            # A skipped deck is logged by the sampler
            candidates = self._sampler.candidate_uuids(
                deck, card_lookup, self._items_per_deck
            )
            if candidates is None:
                continue
            sampled_uuids = self._sampler.full_draw(
                deck, candidates, self._items_per_deck
            )

            start_index = len(items)
            for card_uuid in sampled_uuids:
                items.append(card_for_known_uuid(card_uuid, card_lookup))
                identities.append((card_uuid,))
            positive_cliques.append(list(range(start_index, len(items))))

        return ContrastiveBatch(
            inputs=items, identities=identities, positive_cliques=positive_cliques
        )


class SingleCardInfoNCELoss:
    """Concrete ContrastiveLoss for the single-card slice: single-positive
    InfoNCE over cosine similarity, computed once across the whole flat
    item pool (so every A-B/A-C/B-C deck-pair comparison falls out of
    one pairwise similarity matrix, never special-cased per deck pair).
    Exact-identity duplicates (see ContrastiveBatch.identities) are
    excluded from a given anchor's negative pool at this step, per the
    plan's deferred-to-loss-time policy - never precomputed or stored
    on the batch itself."""

    def __init__(self, temperature: float = 0.07) -> None:
        """
        Inputs:
            temperature: InfoNCE's softmax temperature - lower sharpens
                the similarity distribution. 0.07 is CLIP's commonly
                cited default.
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
        """Mean anchor-wise single-positive InfoNCE loss over the batch.

        Inputs: see ContrastiveLoss.calculate(). item_embeddings must be
            single-card-shaped (one bare Embedding per item).
        Output: a scalar loss tensor - the mean, over every item that
            has at least one positive, of that item's own InfoNCE loss
            (its true positive scored against every valid negative -
            every other item, excluding itself and any exact-identity
            duplicate).
        Side effects: none.
        Exceptions: ValueError if len(item_embeddings) != len(identities),
            if item_embeddings isn't single-card-shaped, or if no item
            in the batch has a positive pair (loss is undefined).

        Example:
            >>> loss_fn = SingleCardInfoNCELoss()
            >>> loss_fn.calculate(item_embeddings, identities, positive_cliques)
        """
        check_item_embeddings(
            item_embeddings, identities, InputShape.SINGLE_CARD, "SingleCardInfoNCELoss"
        )
        # Runtime shape already validated above - narrow the wide
        # ContrastiveLoss.calculate() parameter type back to the
        # concrete shape this class actually handles.
        single_card_embeddings = cast(BatchedSingleCardEmbedding, item_embeddings)

        similarity = pairwise_cosine_similarity(single_card_embeddings)
        valid_negative_mask = identity_negative_mask(identities, similarity.device)
        positives_by_anchor = self._positives_by_anchor(positive_cliques)

        # Average each anchor's own InfoNCE loss - an anchor with no
        # positive in this batch contributes nothing. No item has an
        # "excluded" group of its own in the single-card shape (unlike
        # CrossSliceCardInfoNCELoss's own-item siblings), so none is passed.
        return mean_anchor_loss(
            positives_by_anchor, {}, similarity, valid_negative_mask, self._temperature
        )

    def constant_logit_loss(
        self,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> float:
        """See ContrastiveLoss.constant_logit_loss. The comparison unit is
        the item; there are no extra exclusions.

        Example:
            >>> # 3 decks x 2 items, no duplicates: ln(6 - 2 + 1) = ln 5
            >>> SingleCardInfoNCELoss().constant_logit_loss(
            ...     identities, [[0, 1], [2, 3], [4, 5]])
            1.609...
        """
        result: float
        # Same anchor -> positives expansion calculate() uses
        positives_by_anchor = self._positives_by_anchor(positive_cliques)
        result = mean_constant_logit_loss(identities, positives_by_anchor, {})
        return result

    def _positives_by_anchor(
        self, positive_cliques: list[list[int]]
    ) -> dict[int, list[int]]:
        """Expand positive_cliques into a per-anchor positives mapping.

        Private helper - callers are calculate() and
        constant_logit_loss(). Every index in a
        clique is a positive for every other index in that same clique -
        each clique is a mutually-positive group, not a directed
        relation, so this is a pure expansion, not a derivation from
        any pairwise data.

        Inputs:
            positive_cliques: see calculate().
        Output: a mapping from anchor index to the list of item indices
            that are positives for that anchor. An index whose clique
            has no other member (size-1 clique) is simply absent as a
            key.
        Side effects: none.
        Exceptions: none.
        """
        result: dict[int, list[int]] = {}
        for group in positive_cliques:
            for index in group:
                other_indices = [other for other in group if other != index]
                if other_indices:
                    result[index] = other_indices
        return result


@dataclass(frozen=True)
class SingleCardStyle:
    """Today's style: single cards, every same-deck card a positive.

    items_per_deck: single cards drawn per deck (2 in the catalog).
    """

    items_per_deck: int = 2

    def pair_constructor(
        self, rng_seed: int, staple_subsampling: StapleSubsampling | None
    ) -> ContrastivePairConstructor:
        """See ContrastiveStyle.pair_constructor.

        Example:
            >>> SingleCardStyle().pair_constructor(0, None).cards_per_deck
            2
        """
        return SingleCardPairConstructor(
            self.items_per_deck,
            rng_seed=rng_seed,
            staple_subsampling=staple_subsampling,
        )

    def contrastive_loss(self) -> ContrastiveLoss:
        """See ContrastiveStyle.contrastive_loss."""
        return SingleCardInfoNCELoss()
