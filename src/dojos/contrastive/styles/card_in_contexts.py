"""Card across contexts style (plans/multi_card_contrastive_dojos.md, C).

Per deck, one anchor card placed at the front of several different slices
of its deck; each slice attends within itself, and only the anchor's
output is kept:

    [a1 a2 a3] → a1 (context 1)    [a1 a4 a5] → a1 (context 2)
    [b1 b2 b3] → b1 (context 1)    [b1 b4 b5] → b1 (context 2)
    a1 (context 1): positive a1 (context 2); negatives both b1s

Measures "a card stays recognisably itself in any context".

Primarily an evaluation dojo, not for training: keep it out of training
diets and run it on TEST batches as a probe. As a training signal it has
an easy win (attention that ignores context scores perfectly), and for
SingleCardModel (no attention) it is trivially solved: both copies of the
anchor embed identically. As a probe it measures identity retention,
which is read next to a context-sensitivity measure (src/evaluation/TODO.md).
"""

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
)
from src.dojos.contrastive.pair_constructor import (
    ContrastivePairConstructor,
    SliceSampler,
    consecutive_slices,
    contrastive_batch_from_deck_items,
)
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.dojos.contrastive.styles.single_card import SingleCardInfoNCELoss
from src.schema.card import GenericDeck
from src.schema.type_hints import (
    BatchedModelOutput,
    BatchedMultiCardEmbedding,
    Embedding,
    InputShape,
)

# Where each item's anchor sits; the constructor and loss both read this
_ANCHOR_POSITION = 0


class CardInContextsPairConstructor:
    """Concrete ContrastivePairConstructor for the card-across-contexts
    style. Per deck: pick the anchor card (SliceSampler.held_out_draw),
    draw contexts_per_card x context_size other cards with every copy of
    the anchor left out, and cut them into disjoint contexts. Each item is
    [anchor, *context], the anchor always first; the deck's items are one
    positive clique:

        inputs           = [[a1 a2 a3], [a1 a4 a5], [b1 b2 b3], [b1 b4 b5]]
        positive_cliques = [[0, 1], [2, 3]]

    The anchor sits at _ANCHOR_POSITION (first), where AnchorCardLoss
    reads it. Why a fixed position leaks nothing to the model: see
    ContrastiveBatch's docstring.
    """

    def __init__(
        self,
        context_size: int,
        contexts_per_card: int = 2,
        rng_seed: int | None = None,
        staple_subsampling: StapleSubsampling | None = None,
    ) -> None:
        """
        Inputs:
            context_size: cards around the anchor in each item (>= 1).
            contexts_per_card: items per deck (>= 2: one context has no
                positive).
            rng_seed: seed for every draw. None means non-deterministic.
            staple_subsampling: see SliceSampler.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if context_size < 1 or contexts_per_card < 2.
        """
        if context_size < 1:
            raise ValueError("context_size must be positive")
        if contexts_per_card < 2:
            raise ValueError("contexts_per_card must be at least 2")
        self._context_size = context_size
        self._contexts_per_card = contexts_per_card
        self._sampler = SliceSampler(random.Random(rng_seed), staple_subsampling)

    @property
    def cards_per_deck(self) -> int:
        """The anchor once per context, plus every context's cards."""
        return self._contexts_per_card * (1 + self._context_size)

    def build(
        self, decks: list[GenericDeck], card_lookup: CardLookup
    ) -> ContrastiveBatch:
        """contexts_per_card [anchor, *context] items per usable deck.

        Inputs: see ContrastivePairConstructor.build().
        Output: a ContrastiveBatch of multi-card items, contexts_per_card
            consecutive items per surviving deck (anchor first in each),
            one positive clique per deck. Empty if every deck was skipped.
        Side effects: advances the rng; one log line per skipped deck (see
            SliceSampler).
        Exceptions: RuntimeError if card_lookup stops resolving a uuid it
            just resolved; whatever ContrastiveBatch.__post_init__()
            raises on a bookkeeping bug.

        Example:
            >>> constructor = CardInContextsPairConstructor(context_size=7, rng_seed=0)
            >>> constructor.build(deck_sample, card_lookup).positive_cliques
            [[0, 1], [2, 3], ...]
        """
        deck_items: list[list[tuple[UUID, ...]]] = []

        # Each deck: the anchor in front of each of its disjoint contexts
        for deck in decks:
            anchored_items = self._anchored_items(deck, card_lookup)
            if anchored_items is not None:
                deck_items.append(anchored_items)

        return contrastive_batch_from_deck_items(deck_items, card_lookup)

    def _anchored_items(
        self, deck: GenericDeck, card_lookup: CardLookup
    ) -> list[tuple[UUID, ...]] | None:
        """The deck's items as uuid tuples, each (anchor, *context).

        Private helper - single caller is build(). Card occurrences needed
        (copies count separately):
        candidate_uuids(minimum = 1 + contexts_per_card x context_size),
        then held_out_draw(rest_size = contexts_per_card x context_size),
        then the rest cut into contexts_per_card consecutive contexts.
        Inputs: deck, card_lookup.
        Output: contexts_per_card tuples, or None when the deck can't
            fill them (logged by the sampler).
        Side effects: advances the rng.
        Exceptions: none (a draw the anchor's copies leave short is a
            logged skip, not an error).
        """
        rest_size = self._contexts_per_card * self._context_size
        candidates = self._sampler.candidate_uuids(deck, card_lookup, 1 + rest_size)
        if candidates is None:
            return None
        draw = self._sampler.held_out_draw(deck, candidates, rest_size)
        if draw is None:
            return None
        # The anchor first (_ANCHOR_POSITION), then one disjoint context
        return [
            (draw.held_out, *context)
            for context in consecutive_slices(draw.rest, self._context_size)
        ]


class AnchorCardLoss:
    """Decorator (PATTERNS.md) over a single-card ContrastiveLoss: keeps
    only each multi-card item's first card embedding (the anchor in its
    context), maps each item's identity to that anchor's own (so two decks
    anchored on the same card don't count as each other's negatives), and
    hands both to the inner loss with the cliques unchanged.
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
        """The inner loss over each item's anchor embedding.

        Inputs: see ContrastiveLoss.calculate(); item_embeddings must be
            multi-card-shaped, every item non-empty with its anchor first.
        Output: the inner loss's scalar tensor.
        Side effects: none.
        Exceptions: ValueError on a shape or length mismatch; whatever the
            inner loss raises.

        Example:
            >>> AnchorCardLoss(SingleCardInfoNCELoss()).calculate(
            ...     item_embeddings, identities, [[0, 1], [2, 3]])
        """
        # Validate inputs, then keep only each item's anchor
        check_item_embeddings(
            item_embeddings, identities, InputShape.MULTI_CARD, "AnchorCardLoss"
        )
        anchors = _first_card_embeddings(item_embeddings)
        return self._inner.calculate(
            anchors, _anchor_identities(identities), positive_cliques
        )

    def constant_logit_loss(
        self,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> float:
        """See ContrastiveLoss.constant_logit_loss: the inner loss's, on the
        anchors' identities (as calculate() scores them).

        Example:
            >>> AnchorCardLoss(SingleCardInfoNCELoss()).constant_logit_loss(
            ...     identities, [[0, 1], [2, 3]])
            1.098...
        """
        return self._inner.constant_logit_loss(
            _anchor_identities(identities), positive_cliques
        )


def _first_card_embeddings(item_embeddings: BatchedModelOutput) -> list[Embedding]:
    """Each multi-card item's anchor embedding (_ANCHOR_POSITION).

    Private helper - single caller is AnchorCardLoss.calculate().
    Inputs: item_embeddings (multi-card-shaped, already checked).
    Output: list[Embedding], one per item.
    Side effects: none beyond autograd.
    Exceptions: none expected.
    """
    multi_card_embeddings = cast(BatchedMultiCardEmbedding, item_embeddings)
    return [item[_ANCHOR_POSITION] for item in multi_card_embeddings]


def _anchor_identities(
    identities: list[tuple[UUID, ...]],
) -> list[tuple[UUID, ...]]:
    """Each item's identity narrowed to its anchor: (item[0],).

    Private helper - callers are AnchorCardLoss.calculate() and
    constant_logit_loss().
    Inputs: identities, every tuple non-empty.
    Output: list of 1-tuples, same order.
    Side effects: none.
    Exceptions: ValueError on an empty identity tuple.
    """
    result: list[tuple[UUID, ...]] = []
    for identity in identities:
        if not identity:
            raise ValueError("an item with no cards has no anchor")
        result.append((identity[_ANCHOR_POSITION],))
    return result


@dataclass(frozen=True)
class CardInContextsStyle:
    """The card-across-contexts style: CardInContextsPairConstructor and
    AnchorCardLoss(SingleCardInfoNCELoss()). An evaluation dojo, not for
    training diets (see the module docstring).

    context_size, contexts_per_card: see CardInContextsPairConstructor.
    """

    context_size: int = 7
    contexts_per_card: int = 2

    def pair_constructor(
        self, rng_seed: int, staple_subsampling: StapleSubsampling | None
    ) -> ContrastivePairConstructor:
        """See ContrastiveStyle.pair_constructor.

        Example:
            >>> CardInContextsStyle().pair_constructor(0, None).cards_per_deck
            16
        """
        return CardInContextsPairConstructor(
            self.context_size,
            self.contexts_per_card,
            rng_seed=rng_seed,
            staple_subsampling=staple_subsampling,
        )

    def contrastive_loss(self) -> ContrastiveLoss:
        """See ContrastiveStyle.contrastive_loss."""
        return AnchorCardLoss(SingleCardInfoNCELoss())
