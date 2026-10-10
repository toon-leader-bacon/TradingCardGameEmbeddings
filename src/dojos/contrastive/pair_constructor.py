"""The swappable research surface: raw decks -> a ContrastiveBatch.

See src/dojos/README.md's contrastive section. Strategy (PATTERNS.md) -
ContrastiveDojo owns one of these privately and delegates to it, so a
different positive-pair definition or item shape is a new style in
styles/ (its pair constructor, its loss and its ContrastiveStyle), not a
change to ContrastiveDojo's own constructor shape.

This module holds the Protocol and what every style in styles/ shares
(PRINCIPLES.md section 2 - identical logic, not superficially similar):
`SliceSampler` (a deck's candidate cards, staple thinning, slice draws,
held-out draws and the skip-logging policy), `consecutive_slices`,
`contrastive_batch_from_deck_items` and the card lookups
`known_card_uuids`/`card_for_known_uuid`/`cards_for_known_uuids`.
"""

import logging
import random
from dataclasses import dataclass
from typing import Protocol, Sequence
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.staple_subsampling import StapleSubsampling
from src.schema.card import GenericCard, GenericDeck

_logger = logging.getLogger(__name__)


def known_card_uuids(deck: GenericDeck, card_lookup: CardLookup) -> list[UUID]:
    """The subset of deck.card_nocab_uuids that card_lookup has an entry for,
    minus the game's Unknown sentinel (a stand-in for cards the binder
    lacks, not a card to embed).

    Shared by every style's ContrastivePairConstructor (styles/).

    Inputs:
        deck: the deck whose card uuids to filter.
        card_lookup: used to test whether each uuid has an entry.
    Output: the known uuids, in deck.card_nocab_uuids' order, duplicates
        preserved.
    Side effects: none.
    Exceptions: none.
    """
    unknown_uuid = CardBinder.unknown_card_uuid(deck.source_game)
    return [
        card_uuid
        for card_uuid in deck.card_nocab_uuids
        if card_uuid != unknown_uuid and card_lookup.get_by_uuid(card_uuid) is not None
    ]


def card_for_known_uuid(card_uuid: UUID, card_lookup: CardLookup) -> GenericCard:
    """Look up a uuid known_card_uuids already confirmed card_lookup has.

    Shared by every style's ContrastivePairConstructor (styles/).

    Inputs:
        card_uuid: a uuid drawn from a prior known_card_uuids() result
            for this same card_lookup, moments earlier in the same
            build() call.
        card_lookup: the same card_lookup known_card_uuids() was called
            with.
    Output: the matching GenericCard.
    Side effects: none.
    Exceptions: RuntimeError if card_lookup no longer has an entry for
        card_uuid - an environment invariant violation (card_lookup
        changed answers mid-build()), not expected/dirty-data territory.
        A plain `assert` isn't used since -O strips it, and this
        invariant must hold even in an optimized run.
    """
    card = card_lookup.get_by_uuid(card_uuid)
    if card is None:
        raise RuntimeError(
            f"card_lookup no longer has an entry for {card_uuid}, despite "
            "having one moments earlier in this same build() call"
        )
    return card


def cards_for_known_uuids(
    card_uuids: Sequence[UUID], card_lookup: CardLookup
) -> list[GenericCard]:
    """card_for_known_uuid over a slice, in order.

    Shared by every slice-based style's ContrastivePairConstructor (styles/).
    Inputs: card_uuids (from SliceSampler), card_lookup (the same one).
    Output: list[GenericCard], same order and length.
    Side effects: none.
    Exceptions: RuntimeError - see card_for_known_uuid().
    """
    return [card_for_known_uuid(card_uuid, card_lookup) for card_uuid in card_uuids]


def consecutive_slices(
    card_uuids: Sequence[UUID], slice_size: int
) -> list[tuple[UUID, ...]]:
    """card_uuids cut into consecutive slices of slice_size, in order.

    Shared by the styles that draw several disjoint slices in one draw
    (deck slices, card in contexts).

    Inputs: card_uuids (length a multiple of slice_size), slice_size (> 0).
    Output: len(card_uuids) // slice_size tuples.
    Side effects: none.
    Exceptions: ValueError if slice_size <= 0 or the length isn't a
        multiple of it.

    Example:
        >>> consecutive_slices([a, b, c, d], 2)
        [(a, b), (c, d)]
    """
    if slice_size <= 0 or len(card_uuids) % slice_size:
        raise ValueError(
            f"cannot cut {len(card_uuids)} cards into slices of {slice_size}"
        )
    return [
        tuple(card_uuids[start : start + slice_size])
        for start in range(0, len(card_uuids), slice_size)
    ]


def contrastive_batch_from_deck_items(
    deck_items: list[list[tuple[UUID, ...]]], card_lookup: CardLookup
) -> ContrastiveBatch:
    """Assemble a batch of multi-card items: one inner list per surviving
    deck, each of its uuid tuples one item (in order), and the deck's
    items one positive clique (in the same order, so a style may give
    the order within a clique meaning).

    Shared by every slice-based ContrastivePairConstructor in this
    package: each computes its decks' item uuids, then hands them here.

    Inputs: deck_items (decks with no item are simply absent), card_lookup
        (the one the uuids were drawn through).
    Output: ContrastiveBatch; empty if deck_items is.
    Side effects: none.
    Exceptions: RuntimeError - see card_for_known_uuid(); whatever
        ContrastiveBatch.__post_init__() raises.

    Example:
        >>> contrastive_batch_from_deck_items([[(a1, a2), (a3,)]], lookup).positive_cliques
        [[0, 1]]
    """
    items: list[list[GenericCard]] = []
    identities: list[tuple[UUID, ...]] = []
    positive_cliques: list[list[int]] = []
    for item_uuids_of_deck in deck_items:
        start_index = len(items)
        for item_uuids in item_uuids_of_deck:
            items.append(cards_for_known_uuids(item_uuids, card_lookup))
            identities.append(item_uuids)
        positive_cliques.append(list(range(start_index, len(items))))
    return ContrastiveBatch(
        inputs=items, identities=identities, positive_cliques=positive_cliques
    )


class ContrastivePairConstructor(Protocol):
    """Strategy: decides item shape, items-per-deck, and which items are
    each other's positives, then builds one ContrastiveBatch."""

    def __init__(self, rng_seed: int | None = None) -> None:
        """Every concrete implementation accepts at least this - see
        this class's own docstring. Concrete implementations add their
        own sampling-policy parameters on top (e.g. items_per_deck)."""
        ...

    @property
    def cards_per_deck(self) -> int:
        """Cards one surviving deck contributes to a batch (items per deck
        times cards per item). Lets a dojo size a deck sample to a budget."""
        ...

    def build(
        self, decks: list[GenericDeck], card_lookup: CardLookup
    ) -> ContrastiveBatch:
        """Turn one deck sample into one ContrastiveBatch.

        Inputs:
            decks: one deck sample (e.g. from DeckBoxDealer.training_decks()).
            card_lookup: read-only card resolution for decks' card_nocab_uuids.
        Output: a ContrastiveBatch built from decks.
        Side effects: implementation-defined (expected: logging a
            skipped deck; no mutation of decks or card_lookup).
        Exceptions: implementation-defined.
        """
        ...


class SliceSampler:
    """Seeded draws of card slices from decks, shared by every slice-based
    pair constructor so each style samples a deck the same way: the
    deck's known cards (duplicates kept), thinned by optional staple
    subsampling, then drawn without replacement.

    Owns no RNG of its own: it advances the one its constructor is given,
    so a pair constructor's whole draw sequence comes from one seed.
    """

    def __init__(
        self,
        rng: random.Random,
        staple_subsampling: StapleSubsampling | None = None,
    ) -> None:
        """
        Inputs:
            rng: the owning pair constructor's random stream.
            staple_subsampling: thins each deck's known cards before any
                draw (see SingleCardPairConstructor.__init__); None means
                t = inf and draws nothing extra from rng.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._rng = rng
        self._staple_subsampling = staple_subsampling

    def candidate_uuids(
        self, deck: GenericDeck, card_lookup: CardLookup, minimum: int
    ) -> list[UUID] | None:
        """The deck's cards a slice may draw from: its known card uuids
        (known_card_uuids), after staple subsampling when there is one;
        None when fewer than minimum remain.

        Owns the skip-logging policy for every style, so the two skip
        reasons stay distinct: too few known cards is a data problem
        (warning); too few left after staple subsampling is expected at a
        small t (debug).

        Inputs: deck, card_lookup (the split's visible-card lookup),
            minimum (cards the caller needs, > 0).
        Output: list[UUID] in deck order, duplicates kept, at least
            minimum long; or None (logged).
        Side effects: advances the rng when subsampling; one log line on
            a skip.
        Exceptions: none.

        Example:
            >>> sampler.candidate_uuids(deck, card_lookup, minimum=9)
            [UUID('...'), UUID('...'), ...]
        """
        # Known cards first: too few is a data problem, worth a warning
        known_uuids = known_card_uuids(deck, card_lookup)
        if len(known_uuids) < minimum:
            _logger.warning(
                "Skipping deck %s: %d known card(s), need %d",
                deck.nocab_uuid,
                len(known_uuids),
                minimum,
            )
            return None

        # Then staple subsampling: too few left is expected at a small t
        kept_uuids = self._thinned(known_uuids)
        if len(kept_uuids) < minimum:
            _logger.debug(
                "Skipping deck %s: %d card(s) left after staple subsampling",
                deck.nocab_uuid,
                len(kept_uuids),
            )
            return None
        return kept_uuids

    def _thinned(self, known_uuids: list[UUID]) -> list[UUID]:
        """known_uuids after staple subsampling, or known_uuids itself when
        there is none (no rng draw).

        Private helper - single caller is candidate_uuids().
        Inputs: known_uuids. Output: list[UUID].
        Side effects: advances the rng when subsampling.
        Exceptions: none.
        """
        if self._staple_subsampling is None:
            return known_uuids
        return self._staple_subsampling.kept(known_uuids, self._rng)

    def draw(
        self,
        candidates: Sequence[UUID],
        slice_size: int,
        excluded: frozenset[UUID] = frozenset(),
    ) -> list[UUID] | None:
        """slice_size uuids drawn without replacement from candidates,
        leaving out every occurrence of an excluded uuid.

        Inputs: candidates (from candidate_uuids), slice_size (> 0),
            excluded (uuids no copy of which may appear in the slice).
        Output: list[UUID] of length slice_size, in draw order (a card the
            deck holds twice can appear twice); None if fewer than
            slice_size candidates remain after the exclusion.
        Side effects: advances the rng (only when a slice is drawn).
        Exceptions: ValueError if slice_size <= 0.

        Example:
            >>> sampler.draw(candidates, 8, excluded=frozenset({missing_uuid}))
        """
        # Validate inputs
        if slice_size <= 0:
            raise ValueError("slice_size must be positive")

        # Drop every copy of an excluded card, then check there's enough left
        remaining = [uuid for uuid in candidates if uuid not in excluded]
        if len(remaining) < slice_size:
            return None

        # Draw; with no exclusion this is the same rng.sample call
        # SingleCardPairConstructor makes, so the two stay draw-compatible
        return self._rng.sample(remaining, slice_size)

    def full_draw(
        self, deck: GenericDeck, candidates: Sequence[UUID], slice_size: int
    ) -> list[UUID]:
        """draw() for candidates that candidate_uuids(..., minimum >=
        slice_size) just returned, so the draw cannot come up short.

        Inputs: deck (for the error message), candidates, slice_size (> 0).
        Output: list[UUID] of length slice_size, in draw order.
        Side effects: advances the rng.
        Exceptions: RuntimeError if the draw comes up short anyway (an
            invariant violation, so not a plain assert, which -O strips);
            ValueError if slice_size <= 0.

        Example:
            >>> sampler.full_draw(deck, sampler.candidate_uuids(deck, lookup, 8), 8)
        """
        result = self.draw(candidates, slice_size)
        if result is None:
            raise RuntimeError(
                f"deck {deck.nocab_uuid}: no draw of {slice_size} from "
                f"{len(candidates)} candidates"
            )
        return result

    def held_out_draw(
        self, deck: GenericDeck, candidates: list[UUID], rest_size: int
    ) -> "HeldOutDraw | None":
        """Pick one card (uniform over candidate occurrences), then draw
        rest_size more with every copy of the picked card left out.

        Shared by every style built around one card and the cards around
        it: the missing-card style (the picked card is held out of its
        slice) and the card-across-contexts style (the picked card is the
        anchor, the rest its contexts).

        Inputs: deck (for the skip log), candidates (from
            candidate_uuids), rest_size (> 0).
        Output: HeldOutDraw, or None (logged at debug) when the picked
            card's copies leave fewer than rest_size others.
        Side effects: advances the rng; one log line on a skip.
        Exceptions: ValueError if rest_size <= 0 or candidates is empty.

        Example:
            >>> sampler.held_out_draw(deck, candidates, rest_size=8).held_out
            UUID('...')
        """
        # Validate inputs before touching the rng
        if not candidates:
            raise ValueError("no candidates to pick from")
        if rest_size <= 0:
            raise ValueError("rest_size must be positive")

        # Pick, then draw the rest around it
        held_out = self._rng.choice(candidates)
        rest = self.draw(candidates, rest_size, excluded=frozenset({held_out}))
        if rest is None:
            _logger.debug(
                "Skipping deck %s: too few cards left once every copy of %s is out",
                deck.nocab_uuid,
                held_out,
            )
            return None
        return HeldOutDraw(held_out, tuple(rest))


@dataclass(frozen=True)
class HeldOutDraw:
    """One card picked from a deck, and the cards drawn around it (no copy
    of the picked card among them). See SliceSampler.held_out_draw."""

    held_out: UUID
    rest: tuple[UUID, ...]

    def __post_init__(self) -> None:
        """Enforce the invariant: the picked card is not in the rest.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError if held_out is in rest.
        """
        if self.held_out in self.rest:
            raise ValueError(f"held-out card {self.held_out} is among the rest")
