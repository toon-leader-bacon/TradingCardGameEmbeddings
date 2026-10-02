"""DeckSample: a capped, seeded sample of one game's published decks,
held compactly in numpy so the candidate sampler can look up a deck's
distinct cards, the decks containing a card, and each card's document
frequency without going back to SQLite.

Private to held_out_deck_card/ (single consumer: HeldOutDeckCardMetric
and its CandidateSampler).

Layout (CSR, both directions):
    deck_offsets[d]:deck_offsets[d + 1] slices deck_cards -> the sorted,
        distinct eligible card indices of sampled deck d.
    card_offsets[c]:card_offsets[c + 1] slices card_decks -> the sampled
        deck indices containing card c (the inverse postings).
Memory: one int32 per (deck, distinct card) pair in each direction;
500k MTG decks x ~23 distinct cards is ~90 MB in total.
"""

from dataclasses import dataclass
from typing import Iterator
from uuid import UUID

import numpy as np

from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.game_id import GameId

# A deck needs a target plus at least one other eligible card left as
# context once every copy of the target is removed.
MIN_DISTINCT_ELIGIBLE_CARDS = 2


# eq=False: numpy fields have no single-bool equality
@dataclass(frozen=True, eq=False)
class DeckSample:
    """Sampled decks and their distinct eligible cards, as index arrays.

    Fields:
        deck_uuids: sampled deck d's uuid in the published box.
        card_uuids: card index c's nocab_uuid (only cards that occur in
            at least one sampled deck get an index).
        deck_offsets, deck_cards: deck -> distinct card indices (CSR,
            see the module docstring).
        card_offsets, card_decks: card -> deck indices (CSR).

    Inputs: none (data holder). Output: n/a. Side effects: none.
    Exceptions: none.
    """

    deck_uuids: tuple[UUID, ...]
    card_uuids: tuple[UUID, ...]
    deck_offsets: np.ndarray
    deck_cards: np.ndarray
    card_offsets: np.ndarray
    card_decks: np.ndarray

    @classmethod
    def from_deck_box(
        cls,
        deck_box: DeckBox,
        game: GameId,
        max_decks: int | None,
        seed: int,
        excluded_card: UUID,
    ) -> "DeckSample":
        """Read up to max_decks of game's decks (in seeded random rank
        order) and index their distinct eligible cards.

        Inputs:
            deck_box: the published box; only read.
            game: whose decks.
            max_decks: cap on decks read; None reads every deck.
            seed: DeckBox.uuids_ranked_randomly seed.
            excluded_card: never indexed (the game's Unknown sentinel),
                so it can be neither a target nor a decoy.
        Output: a DeckSample. A deck with fewer than
            MIN_DISTINCT_ELIGIBLE_CARDS distinct eligible cards is left
            out (it does not count toward max_decks).
        Side effects: reads deck_box (one ranked scan, then one
            get_by_uuid per deck).
        Exceptions: none of its own.

        Example:
            >>> sample = DeckSample.from_deck_box(
            ...     box, GameId.GWENT, None, 0, CardBinder.unknown_card_uuid(GameId.GWENT)
            ... )
            >>> sample.deck_count
            60197
        """
        card_index: dict[UUID, int] = {}
        deck_uuids: list[UUID] = []
        per_deck_cards: list[np.ndarray] = []

        # Walk the box in seeded rank order until max_decks decks qualify
        for deck_uuid in _ranked_deck_uuids(deck_box, game, seed):
            if max_decks is not None and len(deck_uuids) >= max_decks:
                break
            deck = deck_box.get_by_uuid(deck_uuid)
            if deck is None:
                continue
            eligible = set(deck.card_nocab_uuids) - {excluded_card}
            if len(eligible) < MIN_DISTINCT_ELIGIBLE_CARDS:
                continue
            deck_uuids.append(deck_uuid)
            per_deck_cards.append(_card_indices(eligible, card_index))

        # Pack both CSR directions
        deck_offsets, deck_cards = _packed_rows(per_deck_cards)
        card_offsets, card_decks = _inverted_postings(
            deck_offsets, deck_cards, len(card_index)
        )
        return cls(
            deck_uuids=tuple(deck_uuids),
            card_uuids=tuple(card_index),
            deck_offsets=deck_offsets,
            deck_cards=deck_cards,
            card_offsets=card_offsets,
            card_decks=card_decks,
        )

    @property
    def deck_count(self) -> int:
        """Number of sampled decks.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return len(self.deck_uuids)

    @property
    def card_count(self) -> int:
        """Number of indexed cards.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return len(self.card_uuids)

    def cards_of(self, deck_index: int) -> np.ndarray:
        """deck_index's sorted distinct eligible card indices (a view).

        Inputs: deck_index (int, 0 <= deck_index < deck_count).
        Output: np.ndarray of int32. Side effects: none.
        Exceptions: IndexError for an out-of-range deck_index.
        """
        if not 0 <= deck_index < self.deck_count:
            raise IndexError(f"deck_index {deck_index} out of range")
        return self.deck_cards[
            self.deck_offsets[deck_index] : self.deck_offsets[deck_index + 1]
        ]

    def decks_containing(self, card_index: int) -> np.ndarray:
        """The sampled deck indices whose distinct cards include
        card_index (a view).

        Inputs: card_index (int, 0 <= card_index < card_count).
        Output: np.ndarray of int32. Side effects: none.
        Exceptions: IndexError for an out-of-range card_index.
        """
        if not 0 <= card_index < self.card_count:
            raise IndexError(f"card_index {card_index} out of range")
        return self.card_decks[
            self.card_offsets[card_index] : self.card_offsets[card_index + 1]
        ]

    def document_frequencies(self) -> np.ndarray:
        """How many sampled decks contain each card, indexed by card.

        Inputs: none. Output: np.ndarray of int64, length card_count.
        Side effects: none. Exceptions: none.
        """
        return np.diff(self.card_offsets)


def _ranked_deck_uuids(deck_box: DeckBox, game: GameId, seed: int) -> Iterator[UUID]:
    """game's deck uuids in DeckBox.uuids_ranked_randomly(game, seed)
    order, lazily.

    Inputs: deck_box, game, seed. Output: Iterator[UUID].
    Side effects: reads deck_box. Exceptions: none.
    """
    for deck_uuid, _, _ in deck_box.uuids_ranked_randomly(game, seed):
        yield deck_uuid


def _card_indices(cards: set[UUID], card_index: dict[UUID, int]) -> np.ndarray:
    """One deck's distinct eligible cards as sorted card indices; a card
    seen for the first time gets the next index.

    Inputs: cards (distinct, already without the excluded card),
        card_index (uuid -> index so far).
    Output: np.ndarray of int32, sorted, no duplicates.
    Side effects: adds unseen cards to card_index (the caller's own
        accumulator, built for this one sample).
    Exceptions: none.
    """
    indices = [card_index.setdefault(card, len(card_index)) for card in cards]
    return np.sort(np.asarray(indices, dtype=np.int32))


def _packed_rows(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ragged rows into CSR (offsets, values).

    Inputs: rows (list of int32 arrays).
    Output: (offsets int64 of length len(rows) + 1, values int32).
    Side effects: none. Exceptions: none.
    """
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    if not rows:
        return offsets, np.zeros(0, dtype=np.int32)
    offsets[1:] = np.cumsum([len(row) for row in rows])
    return offsets, np.concatenate(rows).astype(np.int32, copy=False)


def _inverted_postings(
    deck_offsets: np.ndarray, deck_cards: np.ndarray, card_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Invert deck -> cards CSR into card -> decks CSR.

    Inputs: deck_offsets, deck_cards (as DeckSample), card_count.
    Output: (card_offsets int64 of length card_count + 1, card_decks
        int32, each card's decks ascending).
    Side effects: none. Exceptions: none.
    """
    # Every (deck, card) pair, sorted by card then deck (stable sort
    # keeps the deck order, which is already ascending)
    deck_of_pair = np.repeat(
        np.arange(len(deck_offsets) - 1, dtype=np.int32), np.diff(deck_offsets)
    )
    order = np.argsort(deck_cards, kind="stable")
    card_decks = deck_of_pair[order]
    card_offsets = np.zeros(card_count + 1, dtype=np.int64)
    card_offsets[1:] = np.cumsum(np.bincount(deck_cards, minlength=card_count))
    return card_offsets, card_decks
