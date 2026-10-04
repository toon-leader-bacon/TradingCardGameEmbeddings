"""Per-game keys and decks for the 17lands chunk families that carry a
deck: game_data and replay_data (see their READMEs).

GameKeys is each row's (draft_id, match_number, game_number); ChunkDecks
is every distinct deck in a chunk plus each row's index into them.
build_chunk_decks identifies every row's constructed deck once per
chunk, for each family's chunk parser.

A row's deck is its present deck_<name> columns (count > 0), one card
uuid per present column in header order, exactly the list the row
implementation built per row. Rows are grouped by that present-column
pattern first, so each pattern is hashed with
deck_ids.deck_uuid_from_cards() once per chunk, not once per row per
metric. Two patterns can hash to one deck (two header columns naming
one card), so patterns are then merged by deck id, keeping the earliest
row's deck: the deck the row implementation stored first.

store_chunk_decks() writes a chunk's decks into the family DeckBox; it
lives here rather than on ChunkDecks so the chunk data module stays
free of the storage layer.
"""

from dataclasses import dataclass
from uuid import UUID

import numpy as np
import numpy.typing as npt

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.deck_ids import deck_uuid_from_cards
from src.data_refinement.metrics.seventeenlands.zone_counts import ZoneCounts
from src.schema.card import GenericDeck
from src.schema.game_id import GameId


@dataclass(frozen=True)
class GameKeys:
    """Each row's per-game identifier. Neither game_data nor replay_data
    has a single unique-id column, so a game is the composite (draft_id,
    match_number, game_number), written as three output columns.

    draft_id: shape (rows,), str objects.
    match_number, game_number: shape (rows,).
    """

    draft_id: npt.NDArray[np.object_]
    match_number: npt.NDArray[np.int64]
    game_number: npt.NDArray[np.int64]


@dataclass(frozen=True)
class ChunkDecks:
    """Every distinct deck_<name> deck in a chunk, and which one each row
    played.

    decks: one GenericDeck per distinct deck id, in order of first row.
        Its id is deck_ids.deck_uuid_from_cards over the row's present
        deck columns (a card in two columns appears twice), so ids are
        the row implementation's exactly. Its name and card order are
        its first row's ("<family_label> <draft_id>/<match>/<game> deck",
        e.g. "game_data ..." or "replay_data ...").
    row_deck: shape (rows,); row i played decks[row_deck[i]].
    """

    decks: tuple[GenericDeck, ...]
    row_deck: npt.NDArray[np.intp]

    def __post_init__(self) -> None:
        """Enforce one entry per deck id, and every row naming a deck.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError if two decks share an id, or a row_deck
            entry is outside decks.
        """
        deck_uuids = [deck.nocab_uuid for deck in self.decks]
        if len(set(deck_uuids)) != len(deck_uuids):
            raise ValueError("ChunkDecks holds two decks with the same id")
        if self.row_deck.size and (
            self.row_deck.min() < 0 or self.row_deck.max() >= len(self.decks)
        ):
            raise ValueError(
                f"ChunkDecks.row_deck names a deck outside its {len(self.decks)} decks"
            )

    def row_deck_uuids(self) -> npt.NDArray[np.object_]:
        """Each row's deck uuid, as a str.

        Inputs: none. Output: shape (rows,), str objects.
        Side effects: none. Exceptions: none.

        Example:
            >>> chunk.decks.row_deck_uuids()[0]
            '0d9c...'
        """
        deck_uuids = np.array([str(deck.nocab_uuid) for deck in self.decks], object)
        return deck_uuids[self.row_deck]


def build_chunk_decks(
    deck_zone: ZoneCounts, keys: GameKeys, source_game: GameId, family_label: str
) -> ChunkDecks:
    """Every distinct deck in a chunk's deck zone, plus each row's.

    Inputs:
        deck_zone: the chunk's GameZone.DECK counts.
        keys: the chunk's per-row game ids (name each deck after its
            first row's game).
        source_game: stamped on every GenericDeck.
        family_label: the family named in each deck's name ("game_data",
            "replay_data"), e.g. "game_data <draft>/<match>/<game> deck".
    Output: ChunkDecks, one deck per distinct id, in order of first row.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> decks = build_chunk_decks(zone, keys, GameId.MTG, "game_data")
        >>> decks.decks[decks.row_deck[0]]  # row 0's deck
    """
    present = deck_zone.present()

    # Group rows by present-column pattern, in order of first row
    first_rows, row_pattern = _group_rows_by_pattern(present)

    # One GenericDeck per pattern, named after its first row's game
    pattern_decks = [
        _deck_for_row(deck_zone, present, keys, first_row, source_game, family_label)
        for first_row in first_rows
    ]

    # Patterns hashing to one deck id share that id's earliest deck
    decks, pattern_to_deck = _merge_by_deck_id(pattern_decks)
    return ChunkDecks(decks=decks, row_deck=pattern_to_deck[row_pattern])


def store_chunk_decks(decks: ChunkDecks, deck_box: DeckBox) -> None:
    """Store every deck of a chunk in deck_box, once per deck. A deck
    already in the box keeps its stored entry.

    Inputs: decks (a chunk's), deck_box (the family's metrics-private
        box).
    Output: none.
    Side effects: deck_box.create_if_absent() once per deck.
    Exceptions: DeckBox's batch-wide errors (see deck_box.py).

    Example:
        >>> store_chunk_decks(chunk.decks, deck_box)
    """
    for deck in decks.decks:
        deck_box.create_if_absent(deck)


def _group_rows_by_pattern(
    present: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Group present's rows by identical boolean pattern.

    Inputs: present, shape (rows, columns).
    Output: (first_rows, row_group): first_rows[g] is group g's first
        row index, ascending; row_group[i] is row i's group.
    Side effects: none. Exceptions: none.
    """
    rows = present.shape[0]
    if rows == 0:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)

    # Pack each row's pattern into bytes, so np.unique compares rows
    # cheaply; a zone with no columns packs to one constant byte
    packed = (
        np.packbits(present, axis=1)
        if present.shape[1]
        else np.zeros((rows, 1), np.uint8)
    )
    _, first_rows, inverse = np.unique(
        packed, axis=0, return_index=True, return_inverse=True
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
    present: npt.NDArray[np.bool_],
    keys: GameKeys,
    row: int,
    source_game: GameId,
    family_label: str,
) -> GenericDeck:
    """The GenericDeck row plays: one card uuid per present column, in
    header order, named after row's game.

    Inputs: deck_zone, present (deck_zone.present()), keys, row,
        source_game, family_label (the name's prefix).
    Output: GenericDeck with id deck_uuid_from_cards(its cards).
    Side effects: none. Exceptions: none.
    """
    card_nocab_uuids = [
        deck_zone.card_uuids[column] for column in np.flatnonzero(present[row])
    ]
    return GenericDeck(
        nocab_uuid=deck_uuid_from_cards(card_nocab_uuids),
        source_game=source_game,
        name=(
            f"{family_label} {keys.draft_id[row]}/{keys.match_number[row]}/"
            f"{keys.game_number[row]} deck"
        ),
        card_nocab_uuids=card_nocab_uuids,
    )
