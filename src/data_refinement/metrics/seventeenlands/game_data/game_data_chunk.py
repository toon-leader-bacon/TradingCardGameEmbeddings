"""GameDataChunk - one typed, numpy-backed block of 17lands game_data
rows, the unit every game_data metric's accumulate() receives (see
game_data/README.md).

A chunk is parsed once per record batch by GameDataChunkParser
(game_data_chunk_parser.py) and read by every metric, so no metric ever
sees a CSV column name or a raw row dict. Card columns are grouped per
zone (GameZone) into ZoneCounts: one matrix column per matched header
column, so two header columns that share a card uuid stay two columns,
which preserves the row implementation's per-column counting exactly.

Each row's constructed deck (its deck_<name> columns) is identified
once per chunk, in ChunkDecks, so the deck-input metrics never hash a
deck themselves.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from uuid import UUID

import numpy as np
import numpy.typing as npt

from src.schema.card import GenericDeck


class GameZone(Enum):
    """A game_data card-column family, valued by its header prefix."""

    OPENING_HAND = "opening_hand_"
    DRAWN = "drawn_"
    TUTORED = "tutored_"
    DECK = "deck_"
    SIDEBOARD = "sideboard_"


@dataclass(frozen=True)
class ZoneCounts:
    """One zone's per-game copy counts for a chunk's rows.

    card_uuids: one card uuid per matched header column, in header
        order; may repeat (two columns matching one card).
    counts: shape (rows, len(card_uuids)); a null CSV cell is 0, which
        matches the row implementation's "NaN means absent".
    """

    card_uuids: tuple[UUID, ...]
    counts: npt.NDArray[np.int16]

    def present(self) -> npt.NDArray[np.bool_]:
        """Which (row, column) cells hold at least one copy. (The row
        implementation tested "nonzero"; a count is never negative, so
        the two agree.)

        Inputs: none. Output: bool array shaped like counts.
        Side effects: none. Exceptions: none.

        Example:
            >>> chunk.zones[GameZone.DECK].present().sum(axis=0)
        """
        return self.counts > 0

    def present_for(self, card_uuids: tuple[UUID, ...]) -> npt.NDArray[np.bool_]:
        """Which rows hold each of card_uuids in this zone, under any of
        its columns. Lines this zone up against another zone's columns
        (e.g. "was this deck column's card tutored?").

        Inputs: card_uuids (any cards; may repeat, may include cards
            this zone has no column for).
        Output: bool array, shape (rows, len(card_uuids)); a card with
            no column in this zone is never present.
        Side effects: none. Exceptions: none.

        Example:
            >>> deck = chunk.zones[GameZone.DECK]
            >>> chunk.zones[GameZone.TUTORED].present_for(deck.card_uuids)
        """
        rows = self.counts.shape[0]
        result = np.zeros((rows, len(card_uuids)), dtype=np.bool_)

        # One "any column of this card" vector per distinct card here
        present_per_card = self._present_per_distinct_card()

        # Copy each requested card's vector into its output column
        for output_column, card_uuid in enumerate(card_uuids):
            card_present = present_per_card.get(card_uuid)
            if card_present is not None:
                result[:, output_column] = card_present
        return result

    def _present_per_distinct_card(self) -> dict[UUID, npt.NDArray[np.bool_]]:
        """Per distinct card uuid, whether any of its columns is present,
        per row.

        Inputs: none. Output: dict card uuid -> bool array (rows,).
        Side effects: none. Exceptions: none.
        """
        present = self.present()
        result: dict[UUID, npt.NDArray[np.bool_]] = {}

        # A card's vector is the OR of its columns (usually just one)
        for column, card_uuid in enumerate(self.card_uuids):
            previous = result.get(card_uuid)
            column_present = present[:, column]
            result[card_uuid] = (
                column_present if previous is None else previous | column_present
            )
        return result


@dataclass(frozen=True)
class GameKeys:
    """Each row's per-game identifier. game_data has no single unique-id
    column, so a game is the composite (draft_id, match_number,
    game_number), written as three output columns.

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
        its first row's ("game_data <draft_id>/<match>/<game> deck").
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


@dataclass(frozen=True)
class GameDataChunk:
    """A block of game_data rows, parsed once and shared by every metric.

    zones: every GameZone's ZoneCounts (a zone with no matched columns
        has an empty card_uuids and a (rows, 0) counts matrix).
    won, on_play: shape (rows,); a null in the CSV raises at parse time.
    num_turns: shape (rows,); a null raises at parse time.
    keys: each row's (draft_id, match_number, game_number).
    rank: shape (rows,), str objects; "" where the event has no rank
        (every Trad and Sealed format leaves the column empty).
    decks: each row's deck_<name> deck, identified once for the chunk.
    """

    zones: Mapping[GameZone, ZoneCounts]
    won: npt.NDArray[np.bool_]
    on_play: npt.NDArray[np.bool_]
    num_turns: npt.NDArray[np.int32]
    keys: GameKeys
    rank: npt.NDArray[np.object_]
    decks: ChunkDecks

    def __post_init__(self) -> None:
        """Enforce the chunk's invariants at construction: every
        GameZone has a ZoneCounts, and every per-row field has the same
        row count.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError naming the missing zone or the field
            whose row count differs.
        """
        missing = [zone.name for zone in GameZone if zone not in self.zones]
        if missing:
            raise ValueError(f"GameDataChunk is missing zones {missing}")

        # Every per-row field must agree with won's row count
        rows = len(self)
        mismatched = {name: n for name, n in self._row_counts().items() if n != rows}
        if mismatched:
            raise ValueError(
                f"GameDataChunk fields disagree with won's {rows} rows: {mismatched}"
            )

    def __len__(self) -> int:
        """Row count.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.

        Example:
            >>> len(chunk)
            65536
        """
        return int(self.won.shape[0])

    def _row_counts(self) -> dict[str, int]:
        """Every per-row field's row count, by field name.

        Inputs: none. Output: dict field name -> rows.
        Side effects: none. Exceptions: none.
        """
        result: dict[str, int] = {
            "on_play": int(self.on_play.shape[0]),
            "num_turns": int(self.num_turns.shape[0]),
            "keys.draft_id": int(self.keys.draft_id.shape[0]),
            "keys.match_number": int(self.keys.match_number.shape[0]),
            "keys.game_number": int(self.keys.game_number.shape[0]),
            "rank": int(self.rank.shape[0]),
            "decks.row_deck": int(self.decks.row_deck.shape[0]),
        }
        for zone, zone_counts in self.zones.items():
            result[f"zones[{zone.name}]"] = int(zone_counts.counts.shape[0])
        return result
