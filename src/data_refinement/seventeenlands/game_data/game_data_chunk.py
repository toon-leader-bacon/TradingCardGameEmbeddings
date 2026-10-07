"""GameDataChunk - one typed, numpy-backed block of 17lands game_data
rows, the unit every game_data metric's accumulate() receives (see
game_data/README.md).

A chunk is parsed once per record batch by GameDataChunkParser
(game_data_chunk_parser.py) and read by every metric, so no metric ever
sees a CSV column name or a raw row dict. Card columns are grouped per
zone (GameZone) into ZoneCounts (../zone_counts.py): one matrix column
per matched header column, so two header columns that share a card uuid stay two columns,
which preserves the row implementation's per-column counting exactly.

Each row's constructed deck (its deck_<name> columns) is identified
once per chunk, in ChunkDecks, so the deck-input metrics never hash a
deck themselves.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
import numpy.typing as npt

from src.data_refinement.seventeenlands.chunk_decks import (
    ChunkDecks,
    GameKeys,
)
from src.data_refinement.seventeenlands.zone_counts import ZoneCounts


class GameZone(Enum):
    """A game_data card-column family, valued by its header prefix."""

    OPENING_HAND = "opening_hand_"
    DRAWN = "drawn_"
    TUTORED = "tutored_"
    DECK = "deck_"
    SIDEBOARD = "sideboard_"


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
