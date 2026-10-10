"""Odd-one-out style (plans/multi_card_contrastive_dojos.md, E).

Per deck, a slice with one card from another deck in the batch swapped
in. The slice attends within itself; each card is scored by how well it
fits the rest of the slice, and the intruder must score lowest:

    [a1 a2 a3 b3]   intruder: b3
    [b1 b2 b4 a5]   intruder: a5

The removal-side twin of the missing-card style: "which card doesn't
belong here?" rather than "which card is missing?".
"""

import logging
import math
import random
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import torch

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.contrastive_loss import (
    ContrastiveLoss,
    DegenerateBatchError,
    check_item_embeddings,
)
from src.dojos.contrastive.pair_constructor import (
    ContrastivePairConstructor,
    SliceSampler,
    contrastive_batch_from_deck_items,
    known_card_uuids,
)
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.schema.card import GenericDeck
from src.schema.type_hints import (
    BatchedModelOutput,
    BatchedMultiCardEmbedding,
    Embedding,
    InputShape,
)

_logger = logging.getLogger(__name__)

# Where each item's intruder sits; the constructor and loss both read this
_INTRUDER_POSITION = -1


@dataclass(frozen=True)
class _DeckCandidates:
    """One surviving deck's draw pool: its candidate cards, and every card
    it knows (the intruder must be none of them, not just outside the
    slice)."""

    deck: GenericDeck
    candidates: tuple[UUID, ...]
    known: frozenset[UUID]


class OddOneOutPairConstructor:
    """Concrete ContrastivePairConstructor for the odd-one-out style.

    Per deck: a slice of slice_size - 1 of its own cards, then one
    intruder from another surviving deck in the same batch that the host
    deck doesn't hold at all. Each item is one slice; the deck's clique is
    just that item:

        inputs           = [[a1 a2 a3 b3], [b1 b2 b4 a5]]
        positive_cliques = [[0], [1]]

    The intruder sits at _INTRUDER_POSITION (last), and OddOneOutLoss
    reads it there. Why a fixed position leaks nothing to the model: see
    ContrastiveBatch's docstring.
    """

    def __init__(
        self,
        slice_size: int,
        rng_seed: int | None = None,
        staple_subsampling: StapleSubsampling | None = None,
    ) -> None:
        """
        Inputs:
            slice_size: cards per item, the intruder included (>= 3: with
                two cards neither fits the other better).
            rng_seed: seed for every draw. None means non-deterministic.
            staple_subsampling: see SliceSampler; thins both the host's
                slice cards and the donor's intruder candidates.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if slice_size < 3.
        """
        if slice_size < 3:
            raise ValueError("slice_size must be at least 3")
        self._slice_size = slice_size
        # One stream for everything: the sampler draws from it too, so a
        # seed fixes the thinning, donors, intruders and slices together
        self._rng = random.Random(rng_seed)
        self._sampler = SliceSampler(self._rng, staple_subsampling)

    @property
    def cards_per_deck(self) -> int:
        """The host slice plus the intruder."""
        return self._slice_size

    def build(
        self, decks: list[GenericDeck], card_lookup: CardLookup
    ) -> ContrastiveBatch:
        """One [host slice..., intruder] item per usable deck.

        Inputs: see ContrastivePairConstructor.build().
        Output: a ContrastiveBatch of multi-card items, one per deck that
            got both a slice and an intruder, each its own one-item
            clique. Empty when fewer than two decks survive (there is no
            donor), or every deck was skipped.
        Side effects: advances the rng; one log line per skipped deck.
        Exceptions: RuntimeError if card_lookup stops resolving a uuid it
            just resolved; whatever ContrastiveBatch.__post_init__()
            raises on a bookkeeping bug.

        Example:
            >>> constructor = OddOneOutPairConstructor(slice_size=8, rng_seed=0)
            >>> constructor.build(deck_sample, card_lookup).positive_cliques
            [[0], [1], ...]
        """
        deck_items: list[list[tuple[UUID, ...]]] = []

        # Every surviving deck's pool first: any of them can be a donor
        pools = self._deck_candidates(decks, card_lookup)
        if len(pools) < 2:
            return contrastive_batch_from_deck_items([], card_lookup)

        # Each host: its own slice, then an intruder from another deck,
        # appended so it sits at _INTRUDER_POSITION
        for host_index, host in enumerate(pools):
            intruder = self._intruder_for(host_index, pools)
            if intruder is None:
                continue
            host_slice = self._sampler.full_draw(
                host.deck, host.candidates, self._slice_size - 1
            )
            deck_items.append([(*host_slice, intruder)])

        return contrastive_batch_from_deck_items(deck_items, card_lookup)

    def _deck_candidates(
        self, decks: list[GenericDeck], card_lookup: CardLookup
    ) -> list[_DeckCandidates]:
        """Each deck with enough candidates for its slice, with the set of
        every card it knows.

        Private helper - single caller is build().
        Inputs: decks, card_lookup.
        Output: one _DeckCandidates per surviving deck, in deck order
            (skips logged by the sampler).
        Side effects: advances the rng (staple subsampling).
        Exceptions: none.
        """
        result: list[_DeckCandidates] = []
        for deck in decks:
            candidates = self._sampler.candidate_uuids(
                deck, card_lookup, self._slice_size - 1
            )
            if candidates is None:
                continue
            known = frozenset(known_card_uuids(deck, card_lookup))
            result.append(_DeckCandidates(deck, tuple(candidates), known))
        return result

    def _intruder_for(
        self, host_index: int, pools: list[_DeckCandidates]
    ) -> UUID | None:
        """A card from a random other deck's candidates that the host deck
        doesn't know at all.

        Private helper - single caller is build().
        Inputs: host_index (into pools), pools (at least two).
        Output: the intruder's uuid, or None (logged at debug) when the
            chosen donor has no candidate outside the host deck.
        Side effects: advances the rng; one log line on a skip.
        Exceptions: none.
        """
        host = pools[host_index]
        donor = self._rng.choice(
            [pool for index, pool in enumerate(pools) if index != host_index]
        )
        options = [uuid for uuid in donor.candidates if uuid not in host.known]
        if not options:
            _logger.debug(
                "Skipping deck %s: donor deck %s holds no card it lacks",
                host.deck.nocab_uuid,
                donor.deck.nocab_uuid,
            )
            return None
        return self._rng.choice(options)


class OddOneOutLoss:
    """Concrete ContrastiveLoss for the odd-one-out style. Per item of n
    cards (contextual embeddings e_1 ... e_n): each card's fit is the
    cosine of e_k with the mean of the item's other cards; the logits are
    -fit / temperature, and the loss is the cross-entropy with
    _INTRUDER_POSITION (see OddOneOutPairConstructor) as the target,
    averaged over items. Not InfoNCE: the candidates are positions within
    one item, not other items in the batch, so the module's shared
    InfoNCE helpers don't apply.

    Baseline (equal logits): ln(n) per item, averaged over items.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """
        Inputs:
            temperature: softmax temperature over an item's fit scores
                (lower sharpens). Same default as the InfoNCE losses.
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
        """Mean per-item cross-entropy of finding the intruder.

        Inputs: see ContrastiveLoss.calculate(); item_embeddings must be
            multi-card-shaped, each item at least 3 cards with the
            intruder last; positive_cliques one singleton per item.
        Output: a scalar loss tensor.
        Side effects: none.
        Exceptions: DegenerateBatchError when there are no items;
            ValueError on a shape or length mismatch, a clique that isn't
            a singleton, cliques that don't cover every item once, or an
            item under 3 cards.

        Example:
            >>> OddOneOutLoss().calculate(item_embeddings, identities, [[0], [1]])
        """
        # Validate inputs
        check_item_embeddings(
            item_embeddings, identities, InputShape.MULTI_CARD, "OddOneOutLoss"
        )
        _check_singleton_items(identities, positive_cliques)

        multi_card_embeddings = cast(BatchedMultiCardEmbedding, item_embeddings)

        # Each item: find the last card among all of them
        per_item_losses = [self._item_loss(item) for item in multi_card_embeddings]
        return torch.stack(per_item_losses).mean()

    def constant_logit_loss(
        self,
        identities: list[tuple[UUID, ...]],
        positive_cliques: list[list[int]],
    ) -> float:
        """See ContrastiveLoss.constant_logit_loss: ln(item size), averaged
        over items.

        Example:
            >>> OddOneOutLoss().constant_logit_loss(identities, [[0], [1]])  # 8-card items
            2.079...
        """
        _check_singleton_items(identities, positive_cliques)
        return math.fsum(math.log(len(identity)) for identity in identities) / len(
            identities
        )

    def _item_loss(self, card_embeddings: list[Embedding]) -> torch.Tensor:
        """One item's cross-entropy of the last card having the lowest fit.

        Private helper - single caller is calculate().
        Inputs: card_embeddings, at least 3, intruder last.
        Output: a scalar tensor.
        Side effects: none beyond autograd.
        Exceptions: none expected.
        """
        stacked = torch.nn.functional.normalize(torch.stack(card_embeddings), dim=1)
        card_count = stacked.shape[0]
        # Mean of every OTHER card, per card: (sum - own) / (n - 1)
        others_mean = (stacked.sum(dim=0, keepdim=True) - stacked) / (card_count - 1)
        fit = torch.nn.functional.cosine_similarity(stacked, others_mean, dim=1)
        # The worst fit should win: negate, then softmax over positions
        log_probabilities = torch.log_softmax(-fit / self._temperature, dim=0)
        return -log_probabilities[_INTRUDER_POSITION]


def _check_singleton_items(
    identities: list[tuple[UUID, ...]], positive_cliques: list[list[int]]
) -> None:
    """The odd-one-out batch layout: one singleton clique per item,
    covering every item once, each item at least 3 cards.

    Private helper - callers are OddOneOutLoss.calculate() and
    constant_logit_loss().
    Inputs: identities, positive_cliques.
    Output: none.
    Side effects: none.
    Exceptions: DegenerateBatchError if there are no items (every deck was
        skipped: well-formed, just empty); ValueError on any other
        mismatch with the layout.
    """
    # Every deck skipped: well-formed, just empty
    if not identities and not positive_cliques:
        raise DegenerateBatchError("no item in this batch")

    # One singleton clique per item, covering every item once
    if any(len(clique) != 1 for clique in positive_cliques):
        raise ValueError("an odd-one-out clique holds exactly one item")
    covered = sorted(clique[0] for clique in positive_cliques)
    if covered != list(range(len(identities))):
        raise ValueError("the cliques must cover every item exactly once")

    # Each item: at least 3 cards, its intruder nowhere else in it
    for identity in identities:
        if len(identity) < 3:
            raise ValueError(f"an odd-one-out item needs 3+ cards, got {len(identity)}")
        if identity.count(identity[_INTRUDER_POSITION]) != 1:
            raise ValueError("an item's intruder also appears elsewhere in it")


@dataclass(frozen=True)
class OddOneOutStyle:
    """The odd-one-out style: OddOneOutPairConstructor and OddOneOutLoss.

    slice_size: cards per item, intruder included (>= 3); a deck costs
        slice_size cards of the batch budget.
    """

    slice_size: int = 8

    def pair_constructor(
        self, rng_seed: int, staple_subsampling: StapleSubsampling | None
    ) -> ContrastivePairConstructor:
        """See ContrastiveStyle.pair_constructor.

        Example:
            >>> OddOneOutStyle().pair_constructor(0, None).cards_per_deck
            8
        """
        return OddOneOutPairConstructor(
            self.slice_size, rng_seed=rng_seed, staple_subsampling=staple_subsampling
        )

    def contrastive_loss(self) -> ContrastiveLoss:
        """See ContrastiveStyle.contrastive_loss."""
        return OddOneOutLoss()
