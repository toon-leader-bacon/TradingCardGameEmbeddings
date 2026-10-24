"""Missing-card style: a deck slice picks out the card held out of it."""

import logging
import random
from dataclasses import dataclass
from uuid import UUID

import torch

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.contrastive_loss import (
    ComparisonMasks,
    ContrastiveLoss,
    DegenerateBatchError,
    check_item_embeddings,
    clique_positive_mask,
    constant_logit_baseline,
    info_nce_loss,
    info_nce_masks,
    same_key_mask,
)
from src.dojos.contrastive.pair_constructor import (
    ContrastivePairConstructor,
    HeldOutDraw,
    SliceSampler,
    contrastive_batch_from_deck_items,
)
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.schema.card import GenericDeck
from src.schema.padded_embeddings import PaddedEmbeddings
from src.schema.type_hints import InputShape

_logger = logging.getLogger(__name__)


class MissingCardPairConstructor:
    """Concrete ContrastivePairConstructor for the missing-card style
    (plans/multi_card_contrastive_dojos.md, style D).

    Per deck: pick one card, then a slice of slice_size other cards with
    every copy of the picked card removed. Both become multi-card items,
    the slice as a group and the missing card as a one-card group (so
    forward() embeds it alone, through the same path as
    isolated_embeddings):

        inputs           = [[a1 a2 a3], [a4], [b1 b2 b3], [b4], ...]
        positive_cliques = [[0, 1], [2, 3], ...]   # [context, card]

    Each clique is ordered (context index, then card index):
    MissingCardInfoNCELoss reads that order. A deck with too few
    candidate cards for a pick plus a full slice is skipped (logged).
    """

    def __init__(
        self,
        slice_size: int,
        rng_seed: int | None = None,
        staple_subsampling: StapleSubsampling | None = None,
    ) -> None:
        """
        Inputs:
            slice_size: cards in each context slice (>= 2: a one-card
                "context" is not context, and MissingCardInfoNCELoss
                rejects a context item of fewer than 2 cards).
            rng_seed: seed for every draw. None means non-deterministic.
            staple_subsampling: see SliceSampler; thins a deck's cards
                before both the pick and the slice.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if slice_size < 2.
        """
        if slice_size < 2:
            raise ValueError("slice_size must be at least 2")
        self._slice_size = slice_size
        self._sampler = SliceSampler(random.Random(rng_seed), staple_subsampling)

    @property
    def cards_per_deck(self) -> int:
        """The slice plus the missing card."""
        return self._slice_size + 1

    def build(
        self, decks: list[GenericDeck], card_lookup: CardLookup
    ) -> ContrastiveBatch:
        """One (context slice, missing card) pair of items per usable deck.

        Inputs: see ContrastivePairConstructor.build().
        Output: a ContrastiveBatch of multi-card items, two per surviving
            deck (the slice, then the one-card missing item), one ordered
            [context, card] positive clique per deck. Empty if every deck
            was skipped.
        Side effects: one log line per skipped deck (warning or debug by
            reason; see SliceSampler.candidate_uuids, _missing_card_draw).
        Exceptions: RuntimeError if card_lookup stops resolving a uuid it
            just resolved (see card_for_known_uuid); whatever
            ContrastiveBatch.__post_init__() raises on a bookkeeping bug.

        Example:
            >>> constructor = MissingCardPairConstructor(slice_size=8, rng_seed=0)
            >>> batch = constructor.build(deck_sample, card_lookup)
            >>> batch.positive_cliques[0]
            [0, 1]
        """
        deck_items: list[list[tuple[UUID, ...]]] = []

        # Each deck's pair: the context slice, then the missing card as a
        # one-card item; the clique keeps that order, which the loss reads
        for deck in decks:
            draw = self._missing_card_draw(deck, card_lookup)
            if draw is None:
                continue
            deck_items.append([draw.rest, (draw.held_out,)])

        return contrastive_batch_from_deck_items(deck_items, card_lookup)

    def _missing_card_draw(
        self, deck: GenericDeck, card_lookup: CardLookup
    ) -> HeldOutDraw | None:
        """The missing card (HeldOutDraw.held_out) and its context slice
        (HeldOutDraw.rest). The pick is uniform over the deck's candidate
        occurrences, so a 4-of is 4x as likely as a 1-of; staple
        subsampling is the knob against that.

        Private helper - single caller is build().
        Inputs: deck, card_lookup.
        Output: HeldOutDraw, or None (logged by the sampler) when the deck
            can't fill one.
        Side effects: advances the rng; one log line on a skip.
        Exceptions: none.
        """
        candidates = self._sampler.candidate_uuids(
            deck, card_lookup, self.cards_per_deck
        )
        if candidates is None:
            return None
        return self._sampler.held_out_draw(deck, candidates, self._slice_size)


@dataclass(frozen=True)
class _ContextCardPair:
    """One deck's pair of item indices in a missing-card batch: its
    context slice and the card held out of it."""

    context_index: int
    card_index: int


class MissingCardInfoNCELoss:
    """Concrete ContrastiveLoss for the missing-card style
    (MissingCardPairConstructor; plans/multi_card_contrastive_dojos.md,
    style D). CLIP-style symmetric InfoNCE between context slices and
    lone cards:

                   card_A   card_B
        context_A    pos      neg
        context_B    neg      pos

    The comparison unit is the item: a context slice's unit is the mean
    of its contextualized card embeddings, a missing card's unit is its
    one embedding. Every unit is an anchor. A context's positive is its
    own card and its negatives are the other cards; the other contexts
    are excluded (never compared). The same holds with the roles
    swapped. So the loss reduces to the shared info_nce_loss with
    same-role units excluded. Two decks missing the same card are each
    other's right answers too, so each excludes the other's card and
    context as well (identity duplicates can't catch this: a context and
    a card never share an identity).
    """

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
        """Mean anchor-wise InfoNCE over every context and every card.

        Inputs: see ContrastiveLoss.calculate(). item_embeddings must be
            MULTI_CARD; positive_cliques must be ordered [context, card]
            pairs whose card item holds one card (the
            MissingCardPairConstructor layout).
        Output: a scalar loss tensor: the mean over all 2 x (pair count)
            anchors of each anchor's own InfoNCE loss.
        Side effects: none.
        Exceptions: ValueError on a length mismatch, a non-MULTI_CARD
            shape, or a clique that isn't a [context, one-card item]
            pair; DegenerateBatchError for a batch with no pair.

        Example:
            >>> loss_fn = MissingCardInfoNCELoss()
            >>> loss_fn.calculate(item_embeddings, identities, [[0, 1], [2, 3]])
        """
        # Validate inputs and build the masks from the parsed pairs
        check_item_embeddings(
            item_embeddings, identities, InputShape.MULTI_CARD, "MissingCardInfoNCELoss"
        )
        masks = self._masks(positive_cliques, identities)

        # One unit per item: a context's mean embedding, a card's own
        units = item_embeddings.mean_per_example()
        return info_nce_loss(units, masks, self._temperature)

    def constant_logit_loss(
        self,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> float:
        """See ContrastiveLoss.constant_logit_loss. With P pairs and no
        duplicate cards, every anchor has P - 1 negatives, so the
        baseline is ln(P).

        Example:
            >>> MissingCardInfoNCELoss().constant_logit_loss(
            ...     identities, [[0, 1], [2, 3], [4, 5]])
            1.098...
        """
        return constant_logit_baseline(self._masks(positive_cliques, identities))

    def _masks(
        self, positive_cliques: list[list[int]], identities: list[tuple[UUID, ...]]
    ) -> ComparisonMasks:
        """Each context's positive is its own card and vice versa. Every
        same-role unit is excluded (contexts are never compared with
        contexts, nor cards with cards), and so is the other role of
        every pair missing the same card (a right answer, not a negative).

        Private helper - callers are calculate() and constant_logit_loss().
        Inputs: positive_cliques, identities.
        Output: ComparisonMasks over the items: positive from the cliques;
            excluded = same_key_mask(role) | same_key_mask(missing card);
            built from per-item keys, no per-anchor loop.
        Side effects: none.
        Exceptions: as _context_card_pairs.
        """
        pairs = self._context_card_pairs(positive_cliques, identities)
        positive = clique_positive_mask(positive_cliques, len(identities))
        excluded = same_key_mask(_roles(pairs, len(identities))) | same_key_mask(
            _missing_cards(pairs, identities)
        )
        return info_nce_masks(positive, identities, excluded)

    def _context_card_pairs(
        self,
        positive_cliques: list[list[int]],
        identities: list[tuple[UUID, ...]],
    ) -> list[_ContextCardPair]:
        """Parse positive_cliques into [context, card] pairs.

        Private helper - callers are calculate() and constant_logit_loss().
        Inputs: positive_cliques, identities (to check each item's size).
        Output: one _ContextCardPair per clique, in clique order.
        Side effects: none.
        Exceptions: ValueError if there are no cliques, a clique isn't
            exactly two indices, its context item holds fewer than 2
            cards, its card item doesn't hold exactly one, or the cliques
            don't cover every item exactly once.
        """
        result: list[_ContextCardPair] = []
        # Validate inputs: the pairs must partition the items
        if not positive_cliques:
            # Every deck skipped: well-formed, just empty
            raise DegenerateBatchError("no [context, card] pair in this batch")
        covered = sorted(index for clique in positive_cliques for index in clique)
        if covered != list(range(len(identities))):
            raise ValueError("the cliques must cover every item exactly once")

        # Parse each clique, checking each role's item size
        for clique in positive_cliques:
            if len(clique) != 2:
                raise ValueError(
                    f"a missing-card clique is a [context, card] pair, got {clique}"
                )
            context_index, card_index = clique
            if len(identities[context_index]) < 2:
                raise ValueError(
                    f"context item {context_index} holds fewer than 2 cards"
                )
            if len(identities[card_index]) != 1:
                raise ValueError(f"card item {card_index} must hold exactly one card")
            result.append(_ContextCardPair(context_index, card_index))
        return result


def _roles(pairs: list[_ContextCardPair], item_count: int) -> list[bool]:
    """Each item's role: True for a context slice, False for a card.

    Private helper - single caller is MissingCardInfoNCELoss._masks.
    Inputs: pairs (covering every item once), item_count.
    Output: list[bool], one per item. Side effects: none.
    Exceptions: none.
    """
    result = [False] * item_count
    for pair in pairs:
        result[pair.context_index] = True
    return result


def _missing_cards(
    pairs: list[_ContextCardPair], identities: list[tuple[UUID, ...]]
) -> list[tuple[UUID, ...]]:
    """Each item's pair's missing card, for both the context and the card.

    Private helper - single caller is MissingCardInfoNCELoss._masks.
    Inputs: pairs, identities.
    Output: one key per item: identities[pair.card_index] for both items
        of each pair. Side effects: none. Exceptions: none.
    """
    result: list[tuple[UUID, ...]] = [()] * len(identities)
    for pair in pairs:
        missing = identities[pair.card_index]
        result[pair.context_index] = missing
        result[pair.card_index] = missing
    return result


@dataclass(frozen=True)
class MissingCardStyle:
    """The missing-card style: a deck slice must pick out, among the
    batch's lone cards, the card held out of it (and each lone card its
    slice). See plans/multi_card_contrastive_dojos.md, style D.

    slice_size: cards in each context slice (>= 2); a deck then costs
        slice_size + 1 cards of the batch budget. Padding overhead: the
        one-card item is padded to slice_size, so a deck holds
        2 x slice_size padded slots for slice_size + 1 real cards, under
        2x. ContrastiveDojo budgets real cards, so this overhead is not
        charged (see plans/tensor_native_batches.md, section 3).
    """

    slice_size: int = 8

    def pair_constructor(
        self, rng_seed: int, staple_subsampling: StapleSubsampling | None
    ) -> ContrastivePairConstructor:
        """See ContrastiveStyle.pair_constructor.

        Example:
            >>> MissingCardStyle(slice_size=8).pair_constructor(0, None).cards_per_deck
            9
        """
        return MissingCardPairConstructor(
            self.slice_size, rng_seed=rng_seed, staple_subsampling=staple_subsampling
        )

    def contrastive_loss(self) -> ContrastiveLoss:
        """See ContrastiveStyle.contrastive_loss."""
        return MissingCardInfoNCELoss()
