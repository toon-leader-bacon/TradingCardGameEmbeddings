"""DeckSlicesPairConstructor: several disjoint slices per deck, same-deck
slices mutually positive. Shared by the cross-slice card match style
(cross_slice_card.py, per-card loss) and the slice match style
(slice_match.py, pooled loss): the two differ only in their loss."""

import random
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.pair_constructor import (
    SliceSampler,
    consecutive_slices,
    contrastive_batch_from_deck_items,
)
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.schema.card import GenericDeck


class DeckSlicesPairConstructor:
    """Concrete ContrastivePairConstructor: per deck, slices_per_deck
    disjoint slices of slice_size cards, each a multi-card item, and the
    deck's slices one positive clique:

        inputs           = [[a1 a2 a3], [a4 a5 a6], [b1 b2 b3], [b4 b5 b6]]
        positive_cliques = [[0, 1], [2, 3]]

    Disjoint: one draw of slice_size x slices_per_deck cards, cut into
    slices, so no card occurrence is in two slices. (A card the deck holds
    twice can still land in two slices, as two occurrences.) A deck too
    small for every slice is skipped (SliceSampler logs it).
    """

    def __init__(
        self,
        slice_size: int,
        slices_per_deck: int = 2,
        rng_seed: int | None = None,
        staple_subsampling: StapleSubsampling | None = None,
    ) -> None:
        """
        Inputs:
            slice_size: cards per slice (>= 1).
            slices_per_deck: slices per deck (>= 2: a lone slice has no
                positive).
            rng_seed: seed for every draw. None means non-deterministic.
            staple_subsampling: see SliceSampler.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if slice_size < 1 or slices_per_deck < 2.
        """
        if slice_size < 1:
            raise ValueError("slice_size must be positive")
        if slices_per_deck < 2:
            raise ValueError("slices_per_deck must be at least 2")
        self._slice_size = slice_size
        self._slices_per_deck = slices_per_deck
        self._sampler = SliceSampler(random.Random(rng_seed), staple_subsampling)

    @property
    def cards_per_deck(self) -> int:
        """Every slice's cards."""
        return self._slice_size * self._slices_per_deck

    def build(
        self, decks: list[GenericDeck], card_lookup: CardLookup
    ) -> ContrastiveBatch:
        """slices_per_deck disjoint slices per usable deck.

        Inputs: see ContrastivePairConstructor.build().
        Output: a ContrastiveBatch of multi-card items, slices_per_deck
            consecutive items per surviving deck, one positive clique per
            deck. Empty if every deck was skipped.
        Side effects: advances the rng; one log line per skipped deck (see
            SliceSampler.candidate_uuids).
        Exceptions: RuntimeError if card_lookup stops resolving a uuid it
            just resolved; whatever ContrastiveBatch.__post_init__()
            raises on a bookkeeping bug.

        Example:
            >>> constructor = DeckSlicesPairConstructor(slice_size=4, rng_seed=0)
            >>> constructor.build(deck_sample, card_lookup).positive_cliques
            [[0, 1], [2, 3], ...]
        """
        deck_items: list[list[tuple[UUID, ...]]] = []

        # Each deck: one draw for all its slices, cut into consecutive slices
        for deck in decks:
            slices = self._disjoint_slices(deck, card_lookup)
            if slices is not None:
                deck_items.append(slices)

        return contrastive_batch_from_deck_items(deck_items, card_lookup)

    def _disjoint_slices(
        self, deck: GenericDeck, card_lookup: CardLookup
    ) -> list[tuple[UUID, ...]] | None:
        """The deck's slices_per_deck slices, cut from one draw of
        cards_per_deck candidates.

        Private helper - single caller is build().
        Inputs: deck, card_lookup.
        Output: slices_per_deck tuples of slice_size uuids, or None when
            the deck has too few candidates (already logged).
        Side effects: advances the rng.
        Exceptions: RuntimeError if the draw fails after candidate_uuids
            guaranteed enough cards (an invariant violation).
        """
        candidates = self._sampler.candidate_uuids(
            deck, card_lookup, self.cards_per_deck
        )
        if candidates is None:
            return None
        drawn = self._sampler.full_draw(deck, candidates, self.cards_per_deck)
        return consecutive_slices(drawn, self._slice_size)
