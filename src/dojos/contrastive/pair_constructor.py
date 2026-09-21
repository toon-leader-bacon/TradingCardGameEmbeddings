"""The swappable research surface: raw decks -> a ContrastiveBatch.

See plans/contrastive_dojo.md's "ContrastivePairConstructor (swappable)"
section. Strategy (PATTERNS.md) - ContrastiveDojo owns one of these
privately and delegates to it, so a different positive-pair definition
or item shape (single-card vs. multi-card group) is a new class in this
file, not a change to ContrastiveDojo's own constructor shape.

`_known_card_uuids`/`_card_for_known_uuid` below are module-level, shared
by every concrete implementation in this file (PRINCIPLES.md section 2
- identical logic, not superficially similar), rather than duplicated
per class.
"""

import logging
import random
from typing import Protocol
from uuid import UUID

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.schema.card import GenericCard, GenericDeck

_logger = logging.getLogger(__name__)


def _known_card_uuids(deck: GenericDeck, card_lookup: CardLookup) -> list[UUID]:
    """The subset of deck.card_nocab_uuids that card_lookup has an entry for.

    Shared by every concrete ContrastivePairConstructor in this file.

    Inputs:
        deck: the deck whose card uuids to filter.
        card_lookup: used to test whether each uuid has an entry.
    Output: the known uuids, in deck.card_nocab_uuids' order, duplicates
        preserved.
    Side effects: none.
    Exceptions: none.
    """
    return [
        card_uuid
        for card_uuid in deck.card_nocab_uuids
        if card_lookup.get_by_uuid(card_uuid) is not None
    ]


def _card_for_known_uuid(card_uuid: UUID, card_lookup: CardLookup) -> GenericCard:
    """Look up a uuid _known_card_uuids already confirmed card_lookup has.

    Shared by every concrete ContrastivePairConstructor in this file.

    Inputs:
        card_uuid: a uuid drawn from a prior _known_card_uuids() result
            for this same card_lookup, moments earlier in the same
            build() call.
        card_lookup: the same card_lookup _known_card_uuids() was called
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


class SingleCardPairConstructor:
    """Concrete ContrastivePairConstructor for this slice: single-card
    items only, drawn evenly across the given decks. A deck with fewer
    known cards than items_per_deck is skipped outright (its
    nocab_uuid logged) rather than partially sampled - a sampling
    *count* per deck, never full combinatorial enumeration of a deck's
    possible subsets (see plan)."""

    def __init__(self, items_per_deck: int, rng_seed: int | None = None) -> None:
        """
        Inputs:
            items_per_deck: how many single-card items to sample from
                each deck that has at least this many known cards.
            rng_seed: seed for item sampling. None means
                non-deterministic.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if items_per_deck <= 0.
        """
        if items_per_deck <= 0:
            raise ValueError("items_per_deck must be positive")
        self._items_per_deck = items_per_deck
        self._rng = random.Random(rng_seed)

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
            deck is skipped (too few known cards), items/identities/
            positive_cliques are all empty.
        Side effects: emits one logging.warning() per skipped deck.
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
            known_uuids = _known_card_uuids(deck, card_lookup)
            if len(known_uuids) < self._items_per_deck:
                _logger.warning(
                    "Skipping deck %s: %d known card(s), need %d",
                    deck.nocab_uuid,
                    len(known_uuids),
                    self._items_per_deck,
                )
                continue

            sampled_uuids = self._rng.sample(known_uuids, self._items_per_deck)
            start_index = len(items)
            for card_uuid in sampled_uuids:
                items.append(_card_for_known_uuid(card_uuid, card_lookup))
                identities.append((card_uuid,))
            positive_cliques.append(list(range(start_index, len(items))))

        return ContrastiveBatch(
            inputs=items, identities=identities, positive_cliques=positive_cliques
        )


class MultiCardPairConstructor:
    """Concrete ContrastivePairConstructor for the multi-card slice: each
    item is a fixed-size group of cards_per_item cards, sampled together
    but never pooled into one embedding - MultiCardInfoNCELoss compares
    individual cards within an item, not a whole-item vector (see
    contrastive_loss.py). A deck with fewer known cards than
    cards_per_item is skipped outright (its nocab_uuid logged); a
    surviving deck's items_per_deck items are each sampled independently
    (without replacement within one item, but items from the same deck
    may overlap in card membership - an accepted sampling-noise cost,
    same spirit as SingleCardPairConstructor). Mirrors
    SingleCardPairConstructor's skip-and-log/clique-range bookkeeping,
    one level up (items instead of individual cards)."""

    def __init__(
        self, cards_per_item: int, items_per_deck: int, rng_seed: int | None = None
    ) -> None:
        """
        Inputs:
            cards_per_item: how many distinct known cards make up one
                item.
            items_per_deck: how many items to sample from each deck that
                has at least cards_per_item known cards.
            rng_seed: seed for item/card sampling. None means
                non-deterministic.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if cards_per_item <= 0 or items_per_deck <= 0.
        """
        if cards_per_item <= 0:
            raise ValueError("cards_per_item must be positive")
        if items_per_deck <= 0:
            raise ValueError("items_per_deck must be positive")
        self._cards_per_item = cards_per_item
        self._items_per_deck = items_per_deck
        self._rng = random.Random(rng_seed)

    @property
    def cards_per_deck(self) -> int:
        """cards_per_item cards in each of items_per_deck items."""
        return self._cards_per_item * self._items_per_deck

    def build(
        self, decks: list[GenericDeck], card_lookup: CardLookup
    ) -> ContrastiveBatch:
        """Sample items_per_deck multi-card items from each deck, and
        mark every same-deck item mutually positive.

        Inputs: see ContrastivePairConstructor.build().
        Output: a ContrastiveBatch of multi-card items (each item a
            list[GenericCard] of length cards_per_item, each identity a
            same-order tuple of that item's card uuids - see
            ContrastiveBatch.identities). If every given deck is skipped
            (too few known cards), items/identities/positive_cliques are
            all empty.
        Side effects: emits one logging.warning() per skipped deck.
        Exceptions: RuntimeError if card_lookup stops resolving a uuid
            it had just resolved moments earlier in this same call. Also
            whatever ContrastiveBatch.__post_init__() raises, if this
            method's own bookkeeping produces an inconsistent batch.

        Example:
            >>> constructor = MultiCardPairConstructor(cards_per_item=5, items_per_deck=5)
            >>> batch = constructor.build(deck_sample, card_lookup)
        """
        items: list[list[GenericCard]] = []
        identities: list[tuple[UUID, ...]] = []
        positive_cliques: list[list[int]] = []

        # Sample each deck independently, tracking this deck's ITEM-index
        # range in the flat `items` pool - mirrors SingleCardPairConstructor,
        # one level up (a range of items, not individual cards).
        for deck in decks:
            known_uuids = _known_card_uuids(deck, card_lookup)
            if len(known_uuids) < self._cards_per_item:
                _logger.warning(
                    "Skipping deck %s: %d known card(s), need %d",
                    deck.nocab_uuid,
                    len(known_uuids),
                    self._cards_per_item,
                )
                continue

            start_index = len(items)
            for _ in range(self._items_per_deck):
                sampled_cards = self._sample_one_item(known_uuids, card_lookup)
                items.append(sampled_cards)
                identities.append(tuple(card.nocab_uuid for card in sampled_cards))
            positive_cliques.append(list(range(start_index, len(items))))

        return ContrastiveBatch(
            inputs=items, identities=identities, positive_cliques=positive_cliques
        )

    def _sample_one_item(
        self, known_uuids: list[UUID], card_lookup: CardLookup
    ) -> list[GenericCard]:
        """Sample this item's cards_per_item distinct cards.

        Private helper - single caller is build().

        Inputs:
            known_uuids: this deck's known card uuids (see
                _known_card_uuids()) - already confirmed to have at
                least cards_per_item entries by build().
            card_lookup: the same card_lookup known_uuids was derived
                from, moments earlier in the same build() call.
        Output: cards_per_item distinct GenericCard, in sampled order.
        Side effects: none.
        Exceptions: RuntimeError - see _card_for_known_uuid().
        """
        raise NotImplementedError
