"""GameDataChunk - one typed, numpy-backed block of 17lands game_data
rows, the unit every game_data metric's accumulate() receives (see
plans/seventeenlands_chunk_scan.md).

A chunk is parsed once per record batch by GameDataChunkParser
(game_data_chunk_parser.py) and read by every metric, so no metric ever
sees a CSV column name or a raw row dict. Card columns are grouped per
zone (GameZone) into ZoneCounts: one matrix column per matched header
column, so two header columns that share a card uuid stay two columns,
which preserves the row implementation's per-column counting exactly.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pandas as pd


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
        """Which (row, column) cells hold at least one copy.

        Inputs: none. Output: bool array shaped like counts.
        Side effects: none. Exceptions: none.

        Example:
            >>> chunk.zones[GameZone.DECK].present().sum(axis=0)
        """
        return self.counts > 0


@dataclass(frozen=True)
class GameDataChunk:
    """A block of game_data rows, parsed once and shared by every metric.

    zones: every GameZone's ZoneCounts (a zone with no matched columns
        has an empty card_uuids and a (rows, 0) counts matrix).
    won, on_play: shape (rows,); a null in the CSV raises at parse time.
    num_turns: shape (rows,); a null raises at parse time.
    source_frame: the same rows as a pandas DataFrame, set only when
        the parser was built with keep_source_frame=True (a family that
        still wraps row metrics in RowwiseMetric). Migration-only: it
        goes away once game_data has no row metric left.
    """

    zones: Mapping[GameZone, ZoneCounts]
    won: npt.NDArray[np.bool_]
    on_play: npt.NDArray[np.bool_]
    num_turns: npt.NDArray[np.int32]
    source_frame: pd.DataFrame | None

    def __post_init__(self) -> None:
        """Enforce the chunk's invariants at construction: every
        GameZone has a ZoneCounts, and every array (and source_frame,
        when set) has the same row count.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError naming the missing zone or the field
            whose row count differs.
        """
        missing = [zone.name for zone in GameZone if zone not in self.zones]
        if missing:
            raise ValueError(f"GameDataChunk is missing zones {missing}")

        # Every per-row field must agree with won's row count
        rows = len(self)
        row_counts: dict[str, int] = {
            "on_play": int(self.on_play.shape[0]),
            "num_turns": int(self.num_turns.shape[0]),
        }
        for zone, zone_counts in self.zones.items():
            row_counts[f"zones[{zone.name}]"] = int(zone_counts.counts.shape[0])
        if self.source_frame is not None:
            row_counts["source_frame"] = len(self.source_frame)
        mismatched = {name: n for name, n in row_counts.items() if n != rows}
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


def source_frame_of(chunk: GameDataChunk) -> pd.DataFrame:
    """chunk's source_frame, for RowwiseMetric's frame_of.

    Inputs: chunk. Output: pd.DataFrame.
    Side effects: none.
    Exceptions: ValueError if chunk was parsed without a source frame
        (a row metric registered in a family whose parser was built
        with keep_source_frame=False - a driver wiring bug).

    Example:
        >>> RowwiseMetric(inner, frame_of=source_frame_of)
    """
    if chunk.source_frame is None:
        raise ValueError(
            "GameDataChunk has no source_frame: a row metric is wired into a "
            "family whose parser was built with keep_source_frame=False"
        )
    return chunk.source_frame
