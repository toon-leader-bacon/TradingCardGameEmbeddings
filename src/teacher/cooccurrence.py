"""Counting: one game's selected decks -> a card x card co-occurrence
matrix over the selected cards.

The deck x card incidence matrix X is built straight from the deck
snapshot's CSR arrays, weighted by the copy-count and deck-size rules,
and C = X^T X is summed over chunks of decks (MTG has ~10M decks: one
product over all of them would not fit in memory). Unselected cards are
dropped from every deck before counting, which is the counting form of
the <mask>.
"""

import logging
from dataclasses import dataclass
from typing import Callable
from uuid import UUID

import numpy as np
import scipy.sparse

from src.schema.game_id import GameId
from src.schema.holdout import CardTier
from src.schema.splits import Split
from src.teacher.teacher_inputs import TeacherInputs
from src.teacher.teacher_settings import (
    CooccurrenceOptions,
    CopyCountRule,
    DeckSizeRule,
)

_logger = logging.getLogger(__name__)

# Decks per chunk of the incidence matrix
_DECKS_PER_CHUNK = 200_000


@dataclass(frozen=True, eq=False)
class CooccurrenceCounts:
    """One game's co-occurrence over its selected cards and decks.

    source_game: the game.
    card_uuids: the selected cards, in card snapshot order; row i of every
        array below is card_uuids[i].
    pair_counts: float64 (cards, cards) CSR, symmetric, zero diagonal: the
        weighted co-occurrence.
    deck_counts: int64 (cards,): selected decks holding each card.
    deck_count: selected decks counted (after dropping decks left empty).
    """

    source_game: GameId
    card_uuids: tuple[UUID, ...]
    pair_counts: scipy.sparse.csr_matrix
    deck_counts: np.ndarray
    deck_count: int


def count_cooccurrence(
    inputs: TeacherInputs,
    deck_splits: frozenset[Split],
    card_tiers: frozenset[CardTier],
    options: CooccurrenceOptions,
) -> CooccurrenceCounts:
    """Count how often each pair of selected cards shares a selected deck.

    Inputs:
        inputs: the game's decks with the holdout applied.
        deck_splits: whose decks are counted.
        card_tiers: whose cards are kept (others dropped from each deck).
        options: the copy-count and deck-size rules.
    Output: CooccurrenceCounts.
    Side effects: reads the snapshot's mapped arrays; logs progress.
    Exceptions: ValueError if no card or no deck is selected.

    Example:
        >>> counts = count_cooccurrence(
        ...     inputs, frozenset({Split.TRAIN}), frozenset({CardTier.TRAIN}),
        ...     CooccurrenceOptions(),
        ... )
    """
    # Select the decks and cards, and number the kept cards 0..n-1
    deck_rows = inputs.selected_deck_rows(deck_splits)
    card_mask = inputs.selected_card_mask(card_tiers)
    _require_some_selected(deck_rows, card_mask)
    card_positions = _positions_of(card_mask)
    card_total = int(card_mask.sum())
    pair_counts = scipy.sparse.csr_matrix((card_total, card_total), dtype=np.float64)
    deck_counts = np.zeros(card_total, dtype=np.int64)
    deck_count = 0

    # Sum X^T X over chunks of decks
    for chunk in _chunks_of(deck_rows, _DECKS_PER_CHUNK):
        copies = _copies_of(inputs, chunk, card_positions)
        # Deck counts come from presence alone: weighting never moves them
        deck_counts += np.asarray((copies > 0).sum(axis=0)).ravel()
        deck_count += int(np.count_nonzero(copies.getnnz(axis=1)))
        incidence = _weighted(copies, options)
        pair_counts = pair_counts + (incidence.T @ incidence).tocsr()
        _logger.debug("counted %d of %d decks", deck_count, len(deck_rows))

    _logger.info(
        "%s: %d decks, %d cards, %d nonzero pairs (density %.3f)",
        inputs.decks.source_game.value,
        deck_count,
        card_total,
        pair_counts.nnz,
        pair_counts.nnz / max(card_total * card_total, 1),
    )

    # Assemble the result over the kept cards
    card_uuids = tuple(
        card_uuid
        for card_uuid, kept in zip(inputs.decks.card_snapshot.card_uuids, card_mask)
        if kept
    )
    return CooccurrenceCounts(
        source_game=inputs.decks.source_game,
        card_uuids=card_uuids,
        pair_counts=_without_diagonal(pair_counts),
        deck_counts=deck_counts,
        deck_count=deck_count,
    )


def _require_some_selected(deck_rows: np.ndarray, card_mask: np.ndarray) -> None:
    """Raise ValueError if no deck or no card is selected.

    Private helper - single caller is count_cooccurrence.
    Inputs: deck_rows (np.ndarray int64), card_mask (np.ndarray bool).
    Output: None. Side effects: none.
    Exceptions: ValueError naming what is empty.
    """
    if len(deck_rows) == 0:
        raise ValueError("no deck is in the selected deck splits")
    if not card_mask.any():
        raise ValueError("no card is in the selected card tiers")


def _positions_of(card_mask: np.ndarray) -> np.ndarray:
    """Each card snapshot row's column in the kept-card matrix, -1 if the
    card is dropped.

    Private helper - single caller is count_cooccurrence.
    Inputs: card_mask (bool). Output: np.ndarray int64 (len(card_mask),).
    Side effects: none. Exceptions: none.
    """
    result = np.full(len(card_mask), -1, dtype=np.int64)
    result[card_mask] = np.arange(int(card_mask.sum()), dtype=np.int64)
    return result


def _chunks_of(rows: np.ndarray, size: int) -> list[np.ndarray]:
    """rows cut into consecutive slices of at most size.

    Private helper - single caller is count_cooccurrence.
    Inputs: rows (np.ndarray int64), size (int > 0).
    Output: list[np.ndarray] (views). Side effects: none.
    Exceptions: none.
    """
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def _copies_of(
    inputs: TeacherInputs, deck_rows: np.ndarray, card_positions: np.ndarray
) -> scipy.sparse.csr_matrix:
    """One chunk of decks' stored copy counts: one row per deck, one
    column per kept card, dropped cards removed (a deck left with no kept
    card is an empty row). Gathers the chunk's slices of card_indices /
    card_counts by deck_offsets, vectorised (no per-deck Python loop).

    Private helper - single caller is count_cooccurrence.
    Inputs: inputs (TeacherInputs), deck_rows (np.ndarray int64, the
        chunk), card_positions (np.ndarray int64, _positions_of's).
    Output: float64 CSR (len(deck_rows), kept cards) of copy counts.
    Side effects: reads the snapshot's mapped arrays.
    Exceptions: none.
    """
    snapshot = inputs.decks
    starts = snapshot.deck_offsets[deck_rows].astype(np.int64)
    lengths = snapshot.deck_offsets[deck_rows + 1].astype(np.int64) - starts

    # Every entry of every deck in the chunk, by its position in the CSR arrays
    deck_of_entry = np.repeat(np.arange(len(deck_rows)), lengths)
    first_of_deck = np.cumsum(lengths) - lengths
    entry = np.arange(int(lengths.sum())) - np.repeat(first_of_deck, lengths)
    entry += np.repeat(starts, lengths)

    # Keep the entries of kept cards, in kept-card columns
    columns = card_positions[snapshot.card_indices[entry]]
    kept = columns >= 0
    copies = snapshot.card_counts[entry][kept].astype(np.float64)
    shape = (len(deck_rows), int(card_positions.max()) + 1)
    return scipy.sparse.coo_matrix(
        (copies, (deck_of_entry[kept], columns[kept])), shape=shape
    ).tocsr()


def _weighted(
    copies: scipy.sparse.csr_matrix, options: CooccurrenceOptions
) -> scipy.sparse.csr_matrix:
    """The weighted incidence X: _copy_weights on every entry, then each
    deck row times _deck_scales.

    Private helper - single caller is count_cooccurrence.
    Inputs: copies (float64 CSR, _copies_of's), options.
    Output: float64 CSR, same shape and sparsity (a zero-scaled row may
        hold explicit zeros). Side effects: none. Exceptions: none.
    """
    result = copies.copy()
    result.data = _copy_weights(result.data, options.copy_count_rule)
    scales = _deck_scales(result, options.deck_size_rule)
    return scipy.sparse.csr_matrix(scipy.sparse.diags(scales) @ result)


def _copy_weights(card_counts: np.ndarray, rule: CopyCountRule) -> np.ndarray:
    """Each deck entry's weight from its stored copy count: 1 (PRESENCE),
    the count (COPIES) or 1 + ln(count) (LOG_COPIES). A Strategy-by-table
    over the enum, not an if-chain.

    Private helper - single caller is _weighted.
    Inputs: card_counts (np.ndarray float64, stored copies), rule.
    Output: float64, same shape.
    Side effects: none. Exceptions: none.
    """
    return _COPY_WEIGHTS[rule](card_counts)


def _deck_scales(incidence: scipy.sparse.csr_matrix, rule: DeckSizeRule) -> np.ndarray:
    """Each deck row's multiplier: 1 (UNIFORM), or 1 / sqrt(the row's
    weighted pair total, (sum x)^2 - sum x^2) so the deck's pairs in X^T X
    sum to one (INVERSE_SIZE; a deck with no pair scales by 0).

    Private helper - single caller is _weighted.
    Inputs: incidence (float64 CSR, the chunk, copy-weighted), rule.
    Output: float64 (rows,). Side effects: none. Exceptions: none.
    """
    return _DECK_SCALES[rule](incidence)


def _without_diagonal(matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """matrix with its diagonal zeroed and the zeros removed.

    Private helper - single caller is count_cooccurrence.
    Inputs: matrix (square float64 CSR). Output: float64 CSR. Side effects: none.
    Exceptions: none.
    """
    result = scipy.sparse.csr_matrix(matrix - scipy.sparse.diags(matrix.diagonal()))
    result.eliminate_zeros()
    return result


def _inverse_size_scales(incidence: scipy.sparse.csr_matrix) -> np.ndarray:
    """INVERSE_SIZE's row multipliers (see _deck_scales).

    Private helper - single caller is the _DECK_SCALES table.
    Inputs: incidence (float64 CSR). Output: float64 (rows,).
    Side effects: none. Exceptions: none.
    """
    row_sums = np.asarray(incidence.sum(axis=1)).ravel()
    row_squares = np.asarray(incidence.multiply(incidence).sum(axis=1)).ravel()
    pair_totals = row_sums**2 - row_squares
    result = np.zeros(len(pair_totals), dtype=np.float64)
    has_pairs = pair_totals > 0
    result[has_pairs] = 1.0 / np.sqrt(pair_totals[has_pairs])
    return result


# Each copy-count rule's entry weight, from the stored copy counts
_COPY_WEIGHTS: dict[CopyCountRule, Callable[[np.ndarray], np.ndarray]] = {
    CopyCountRule.PRESENCE: np.ones_like,
    CopyCountRule.COPIES: lambda counts: counts,
    CopyCountRule.LOG_COPIES: lambda counts: 1.0 + np.log(counts),
}

# Each deck-size rule's row multipliers
_DECK_SCALES: dict[DeckSizeRule, Callable[[scipy.sparse.csr_matrix], np.ndarray]] = {
    DeckSizeRule.UNIFORM: lambda incidence: np.ones(incidence.shape[0]),
    DeckSizeRule.INVERSE_SIZE: _inverse_size_scales,
}
