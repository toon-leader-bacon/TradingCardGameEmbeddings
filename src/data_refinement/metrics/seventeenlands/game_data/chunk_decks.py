"""build_chunk_decks - identifies every row's constructed deck once per
chunk (see game_data/README.md), for
GameDataChunkParser.

A row's deck is the FULL multiset of its deck_<name> columns: each
present column (count > 0) contributes deck_zone.counts[row, column]
copies of that column's card_uuid, in header order - the same
full-copy-count expansion
deck_box/seventeenlands_game_data/extraction_stage.py's
_card_nocab_uuids_for_row() does for the canonical DeckBox, so two rows
that differ only in copy counts hash to different deck ids. Rows are
grouped by that per-column count pattern first (not merely which
columns are present - see _group_rows_by_pattern()'s own docstring for
why presence alone is no longer a safe coarsening), so each pattern is
hashed with deck_ids.deck_uuid_from_cards() once per chunk, not once
per row per metric. Two patterns can hash to one deck (two header
columns naming one card, or - now that grouping is by count - any
other coincidence that still produces the same multiset), so patterns
are then merged by deck id, keeping the earliest row's deck: the deck
the row implementation stored first.

"""

from uuid import UUID

import numpy as np
import numpy.typing as npt

from src.data_refinement.deck_ids import deck_uuid_from_cards
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    ChunkDecks,
    GameKeys,
    ZoneCounts,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId


def build_chunk_decks(
    deck_zone: ZoneCounts, keys: GameKeys, source_game: GameId
) -> ChunkDecks:
    """Every distinct deck in a chunk's deck zone, plus each row's.

    Inputs:
        deck_zone: the chunk's GameZone.DECK counts.
        keys: the chunk's per-row game ids (name each deck after its
            first row's game).
        source_game: stamped on every GenericDeck.
    Output: ChunkDecks, one deck per distinct id, in order of first row.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> decks = build_chunk_decks(zones[GameZone.DECK], keys, GameId.MTG)
        >>> decks.decks[decks.row_deck[0]]  # row 0's deck
    """
    # Group rows by full copy-count pattern, in order of first row (see
    # _group_rows_by_pattern()'s own docstring for why count, not mere
    # presence, is the safe coarsening once decks are multisets)
    first_rows, row_pattern = _group_rows_by_pattern(deck_zone.counts)

    # One GenericDeck per pattern, named after its first row's game
    pattern_decks = [
        _deck_for_row(deck_zone, keys, first_row, source_game)
        for first_row in first_rows
    ]

    # Patterns hashing to one deck id share that id's earliest deck
    decks, pattern_to_deck = _merge_by_deck_id(pattern_decks)
    return ChunkDecks(decks=decks, row_deck=pattern_to_deck[row_pattern])


def _group_rows_by_pattern(
    counts: npt.NDArray[np.int16],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Group counts's rows by identical per-column copy-count pattern.

    Grouping by mere column PRESENCE (count > 0) is no longer a safe
    coarsening now that a row's deck is the full copy-count multiset
    (see module docstring): two rows can share a present-column pattern
    while differing in copy counts (e.g. one copy of a card vs. two),
    and those hash to different deck ids. Grouping by the full count
    pattern instead keeps every row that pre-groups together on a path
    to the SAME hash, so no two rows with different multisets are ever
    merged before _merge_by_deck_id() runs.

    Inputs: counts, shape (rows, columns).
    Output: (first_rows, row_group): first_rows[g] is group g's first
        row index, ascending; row_group[i] is row i's group.
    Side effects: none. Exceptions: none.
    """
    rows = counts.shape[0]
    if rows == 0:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)

    # A zone with no columns gives every row one constant pattern;
    # np.unique compares rows directly (not packed - a count pattern
    # isn't boolean, so packbits doesn't apply here)
    comparable = counts if counts.shape[1] else np.zeros((rows, 1), np.int16)
    _, first_rows, inverse = np.unique(
        comparable, axis=0, return_index=True, return_inverse=True
    )

    # np.unique orders groups by pattern; reorder them by first row
    by_first_row = np.argsort(first_rows, kind="stable")
    group_rank = np.empty_like(by_first_row)
    group_rank[by_first_row] = np.arange(by_first_row.size)
    row_group = group_rank[inverse.reshape(-1)]
    return first_rows[by_first_row].astype(np.intp), row_group.astype(np.intp)


def _merge_by_deck_id(
    pattern_decks: list[GenericDeck],
) -> tuple[tuple[GenericDeck, ...], npt.NDArray[np.intp]]:
    """Keep the first deck per id, in order.

    Inputs: pattern_decks, one per pattern, in order of first row.
    Output: (decks, pattern_to_deck): the first deck of each id, in
        order; pattern_to_deck[p] is pattern p's index into decks.
    Side effects: none. Exceptions: none.
    """
    decks: list[GenericDeck] = []
    deck_index: dict[UUID, int] = {}
    pattern_to_deck = np.empty(len(pattern_decks), np.intp)

    # Patterns are in first-row order, so each id's first is its earliest
    for pattern, deck in enumerate(pattern_decks):
        index = deck_index.get(deck.nocab_uuid)
        if index is None:
            index = len(decks)
            deck_index[deck.nocab_uuid] = index
            decks.append(deck)
        pattern_to_deck[pattern] = index
    return tuple(decks), pattern_to_deck


def _deck_for_row(
    deck_zone: ZoneCounts,
    keys: GameKeys,
    row: int,
    source_game: GameId,
) -> GenericDeck:
    """The GenericDeck row plays: the full copy-count multiset over
    deck_zone's columns, in header order, named after row's game.

    Each present column (deck_zone.counts[row, column] > 0) contributes
    that many copies of its card_uuid - not merely one per present
    column - so the result is row's genuine full decklist, matching
    deck_box/seventeenlands_game_data/extraction_stage.py's
    _card_nocab_uuids_for_row() expansion.

    Inputs: deck_zone, keys, row, source_game.
    Output: GenericDeck with id deck_uuid_from_cards(its cards).
    Side effects: none. Exceptions: none.

    Example:
        >>> _deck_for_row(deck_zone, keys, 0, GameId.MTG).card_nocab_uuids
        [owlbear, owlbear]  # two copies of Owlbear, none of anything else
    """
    row_counts = deck_zone.counts[row]
    card_nocab_uuids = [
        card_uuid
        for column, card_uuid in enumerate(deck_zone.card_uuids)
        for _ in range(int(row_counts[column]))
    ]
    return GenericDeck(
        nocab_uuid=deck_uuid_from_cards(card_nocab_uuids),
        source_game=source_game,
        name=(
            f"game_data {keys.draft_id[row]}/{keys.match_number[row]}/"
            f"{keys.game_number[row]} deck"
        ),
        card_nocab_uuids=card_nocab_uuids,
    )
