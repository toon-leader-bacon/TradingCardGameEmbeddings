"""DraftDataChunk - one typed, numpy-backed block of 17lands draft_data
rows, the unit every draft_data metric's accumulate() receives (see
draft_data/README.md).

One row is one pick: the pack the drafter saw (pack_card_<name> counts),
the pool drafted so far (pool_<name> counts) and the card(s) taken. No
metric needs another row of the same draft, so a chunk boundary inside
a draft changes nothing.

A chunk is parsed once per record batch by DraftDataChunkParser
(draft_data_chunk_parser.py) and read by every metric, so no metric ever
sees a CSV column name or a raw row dict.

PICK CODES: every matched pack column's card gets a small int code (the
parser's code table), and each row's pick(s) are coded the same way, so
"which pack cell was taken" is an int comparison over the whole chunk
(DraftDataChunk.picked()), never a per-row uuid lookup.
"""

from dataclasses import dataclass
from functools import cached_property
from enum import Enum

import numpy as np
import numpy.typing as npt

from src.data_refinement.seventeenlands.zone_counts import ZoneCounts

# The code of a pick that names no pack column's card (or is empty)
NO_PICK = -1


class DraftStratum(Enum):
    """A per-row value a take rate can be stratified by, valued by its
    output (and CSV) column name."""

    PACK_NUMBER = "pack_number"
    PICK_NUMBER = "pick_number"
    RANK = "rank"


@dataclass(frozen=True)
class DraftPicks:
    """Each row's taken card(s).

    is_pick_two: whether the row took two cards: its `pick_2` cell is
        non-empty (all False in a format without pick_2). Read from the
        cell itself, so an unmatched second card still marks the row.
    first_codes: the `pick` cell's card code (DraftDataChunk's code
        table), NO_PICK if its name matched no pack column's card.
    second_codes: the `pick_2` cell's code; NO_PICK where pick_2 is
        empty, absent or matched no pack column's card. Only for
        DraftDataChunk.picked(); is_pick_two says whether one was taken.
    first_uuids: the `pick` cell's matched nocab_uuid as a str, None if
        its name matched no card (the row streams write it as is).
        Invariant: a None uuid always has code NO_PICK (a matched card
        no pack column names has a uuid but code NO_PICK).
    """

    is_pick_two: npt.NDArray[np.bool_]
    first_codes: npt.NDArray[np.int32]
    second_codes: npt.NDArray[np.int32]
    first_uuids: npt.NDArray[np.object_]


@dataclass(frozen=True)
class DraftDataChunk:
    """One record batch of draft_data rows.

    pack: pack_card_<name> counts, one column per matched header column.
    pool: pool_<name> counts, likewise.
    pack_column_codes: each pack column's card code, shape
        (len(pack.card_uuids),); two columns naming one card share a
        code.
    picks: each row's taken card(s), coded against pack_column_codes.
    draft_id: str per row.
    pack_number, pick_number: int64 per row.
    rank: str per row; "" when the event is unranked.
    """

    pack: ZoneCounts
    pool: ZoneCounts
    pack_column_codes: npt.NDArray[np.int32]
    picks: DraftPicks
    draft_id: npt.NDArray[np.object_]
    pack_number: npt.NDArray[np.int64]
    pick_number: npt.NDArray[np.int64]
    rank: npt.NDArray[np.object_]

    def __post_init__(self) -> None:
        """Reject a chunk whose per-row fields disagree on row count, or
        whose pack codes don't line up with the pack's columns.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError naming the mismatched field.
        """
        rows = self.pack_number.shape[0]
        per_row = {
            "pack.counts": self.pack.counts.shape[0],
            "pool.counts": self.pool.counts.shape[0],
            "picks.is_pick_two": self.picks.is_pick_two.shape[0],
            "picks.first_codes": self.picks.first_codes.shape[0],
            "picks.second_codes": self.picks.second_codes.shape[0],
            "picks.first_uuids": self.picks.first_uuids.shape[0],
            "draft_id": self.draft_id.shape[0],
            "pick_number": self.pick_number.shape[0],
            "rank": self.rank.shape[0],
        }
        for field, length in per_row.items():
            if length != rows:
                raise ValueError(
                    f"DraftDataChunk.{field} has {length} rows, not {rows}"
                )
        if self.pack_column_codes.shape != (len(self.pack.card_uuids),):
            raise ValueError(
                f"DraftDataChunk.pack_column_codes has shape "
                f"{self.pack_column_codes.shape}, not ({len(self.pack.card_uuids)},)"
            )

    def __len__(self) -> int:
        """The chunk's row count.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return int(self.pack_number.shape[0])

    def picked(self) -> npt.NDArray[np.bool_]:
        """Which (row, pack column) cells were taken: present in the pack
        and coded as the row's first or second pick.

        Inputs: none.
        Output: bool array shaped like pack.counts.
        Side effects: none. Exceptions: none.

        Example:
            >>> chunk.picked().sum(axis=0)  # times each column was taken
        """
        return self._picked_cells

    @cached_property
    def _picked_cells(self) -> npt.NDArray[np.bool_]:
        """picked()'s mask, computed once per chunk: every take-rate
        metric reads it.

        Inputs: none. Output: bool array shaped like pack.counts.
        Side effects: caches the array on this instance, read-only (it is
            shared by every metric). Exceptions: none.
        """
        codes = self.pack_column_codes[np.newaxis, :]
        taken = (codes == self.picks.first_codes[:, np.newaxis]) | (
            codes == self.picks.second_codes[:, np.newaxis]
        )
        result = self.pack.present() & taken
        result.flags.writeable = False
        return result

    def stratum(self, stratum: DraftStratum) -> npt.NDArray[np.generic]:
        """One stratifying value per row.

        Inputs: stratum. Output: pack_number / pick_number (int64) or
            rank (str) array, shape (rows,).
        Side effects: none. Exceptions: none.

        Example:
            >>> chunk.stratum(DraftStratum.PICK_NUMBER)
        """
        by_stratum: dict[DraftStratum, npt.NDArray[np.generic]] = {
            DraftStratum.PACK_NUMBER: self.pack_number,
            DraftStratum.PICK_NUMBER: self.pick_number,
            DraftStratum.RANK: self.rank,
        }
        return by_stratum[stratum]
