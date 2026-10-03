"""DraftDataChunkParser - one draft_data CSV's column layout, turning
each pyarrow record batch into one DraftDataChunk (see
draft_data/README.md). A ChunkParser (../chunk_scanner.py).

Card matching is DraftCardColumns' (pack_pool_columns.py), done once per
CSV from its header; pick cell values are matched through the same
DraftCardColumns cache, one lookup per distinct name per chunk.
"""

from collections.abc import Mapping, Sequence
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk import (
    NO_PICK,
    DraftDataChunk,
    DraftPicks,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pack_pool_columns import (
    DraftCardColumns,
)
from src.data_refinement.metrics.seventeenlands.batch_columns import (
    CARD_COUNT_TYPE,
    raise_on_null,
    read_column,
    read_zone_counts,
)
from src.schema.game_id import GameId

# Every scalar column a DraftDataChunk is built from, with its read type.
# A string column reads an empty cell as "" (an unranked event's rank).
_SCALAR_TYPES: Mapping[str, pa.DataType] = {
    "draft_id": pa.string(),
    "pack_number": pa.int64(),
    "pick_number": pa.int64(),
    "pick": pa.string(),
    "rank": pa.string(),
}
# PickTwo formats only: the second card taken
_SECOND_PICK_COLUMN = "pick_2"


class DraftDataChunkParser:
    """One CSV's column layout: matched pack and pool columns, the pack
    card code table, and the scalar columns."""

    def __init__(
        self,
        header: tuple[str, ...],
        draft_columns: DraftCardColumns,
    ) -> None:
        """Use from_header(); this constructor only stores parsed state
        and builds the pack card code table.

        Inputs:
            header: the CSV's full column list, in file order.
            draft_columns: the header's matched pack/pool columns, and
                the name cache pick cells are matched through.
        Output: none (constructor). Side effects: none. Exceptions: none.
        """
        self._header = header
        self._draft_columns = draft_columns
        self._card_codes = _card_codes(draft_columns.pack_columns)
        self._pack_column_codes = np.array(
            [self._card_codes[uuid] for _, uuid in draft_columns.pack_columns],
            np.int32,
        )
        self._has_second_pick = _SECOND_PICK_COLUMN in header

    @classmethod
    def from_header(
        cls, header: Sequence[str], card_binder: CardBinder, source_game: GameId
    ) -> "DraftDataChunkParser":
        """Match one CSV's header against card_binder (Factory Method).

        Inputs: header (the CSV's column names), card_binder (populated
            for source_game), source_game.
        Output: a DraftDataChunkParser.
        Side effects: none (no I/O).
        Exceptions: ValueError if a column in _SCALAR_TYPES is missing.

        Example:
            >>> DraftDataChunkParser.from_header(header, binder, GameId.MTG)
        """
        missing = [column for column in _SCALAR_TYPES if column not in header]
        if missing:
            raise ValueError(f"draft_data CSV header lacks scalar columns {missing}")
        draft_columns = DraftCardColumns.from_header(header, card_binder, source_game)
        return cls(tuple(header), draft_columns)

    def needed_columns(self) -> list[str]:
        """The columns the scanner must read: matched pack and pool
        columns, the scalars, and pick_2 where present.

        Inputs: none. Output: list[str], header order.
        Side effects: none. Exceptions: none.

        Example:
            >>> pyarrow.csv.ConvertOptions(include_columns=parser.needed_columns())
        """
        wanted = {column for column, _ in self._draft_columns.pack_columns}
        wanted.update(column for column, _ in self._draft_columns.pool_columns)
        wanted.update(_SCALAR_TYPES)
        wanted.add(_SECOND_PICK_COLUMN)
        return [column for column in self._header if column in wanted]

    def column_types(self) -> dict[str, pa.DataType]:
        """Read types: every card column as float32 (some exports write
        "1.0"), each scalar as _SCALAR_TYPES says, pick_2 as a string.

        Inputs: none. Output: dict column -> pyarrow type.
        Side effects: none. Exceptions: none.

        Example:
            >>> pyarrow.csv.ConvertOptions(column_types=parser.column_types())
        """
        result: dict[str, pa.DataType] = dict(_SCALAR_TYPES)
        if self._has_second_pick:
            result[_SECOND_PICK_COLUMN] = pa.string()
        for columns in (
            self._draft_columns.pack_columns,
            self._draft_columns.pool_columns,
        ):
            for column, _ in columns:
                result[column] = CARD_COUNT_TYPE
        return result

    def parse(self, batch: pa.RecordBatch) -> DraftDataChunk:
        """Turn one record batch into a DraftDataChunk.

        Inputs: batch, read with needed_columns()/column_types().
        Output: DraftDataChunk.
        Side effects: may add pick names to the DraftCardColumns cache.
        Exceptions: ValueError naming the column and first row offset if
            a scalar column holds a null.

        Example:
            >>> chunk = parser.parse(next(iter(reader)))
        """
        # Validate the scalars before building anything from them
        for column in _SCALAR_TYPES:
            raise_on_null(batch, column)

        # The pack and the pool: one count matrix each
        pack = read_zone_counts(batch, self._draft_columns.pack_columns)
        pool = read_zone_counts(batch, self._draft_columns.pool_columns)

        return DraftDataChunk(
            pack=pack,
            pool=pool,
            pack_column_codes=self._pack_column_codes,
            picks=self._picks(batch),
            draft_id=read_column(batch, "draft_id", np.object_),
            pack_number=read_column(batch, "pack_number", np.int64),
            pick_number=read_column(batch, "pick_number", np.int64),
            rank=read_column(batch, "rank", np.object_),
        )

    def _picks(self, batch: pa.RecordBatch) -> DraftPicks:
        """Each row's pick(s), coded against the pack code table.

        Inputs: batch. Output: DraftPicks (second_codes all NO_PICK when
            the CSV has no pick_2 column).
        Side effects: may add pick names to the DraftCardColumns cache.
        Exceptions: none.
        """
        first_names = read_column(batch, "pick", np.object_)
        first_codes, first_uuids = self._code_names(first_names)
        if not self._has_second_pick:
            no_second = np.full(batch.num_rows, NO_PICK, np.int32)
            return DraftPicks(
                np.zeros(batch.num_rows, np.bool_), first_codes, no_second, first_uuids
            )

        # A PickTwo row: its pick_2 cell is non-empty, matched or not
        second_names = read_column(batch, _SECOND_PICK_COLUMN, np.object_)
        is_pick_two = _non_empty(second_names)
        second_codes, _ = self._code_names(second_names)
        return DraftPicks(is_pick_two, first_codes, second_codes, first_uuids)

    def _code_names(
        self, names: npt.NDArray[np.object_]
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.object_]]:
        """Card names as (code, uuid str) per row: each distinct name is
        matched once; an empty, null or unmatched name codes as NO_PICK
        with a None uuid, and a matched card no pack column names codes
        as NO_PICK with its uuid. (`pick` is never null - parse() rejects
        that - but pick_2 is empty on single picks.)

        Inputs: names (object array of str or None).
        Output: (int32 codes, object array of str or None).
        Side effects: may add names to the DraftCardColumns cache.
        Exceptions: none.
        """
        codes = np.full(names.shape[0], NO_PICK, np.int32)
        uuids = np.full(names.shape[0], None, np.object_)
        present = _non_empty(names)
        if not present.any():
            return codes, uuids

        # Match each distinct name once, then scatter to its rows
        distinct, row_name = np.unique(names[present].astype(str), return_inverse=True)
        name_uuids = [self._draft_columns.uuid_for_name(name) for name in distinct]
        name_codes = np.array(
            [
                NO_PICK if u is None else self._card_codes.get(u, NO_PICK)
                for u in name_uuids
            ],
            np.int32,
        )
        name_strings = np.array(
            [None if u is None else str(u) for u in name_uuids], np.object_
        )
        codes[present] = name_codes[row_name]
        uuids[present] = name_strings[row_name]
        return codes, uuids


def _non_empty(names: npt.NDArray[np.object_]) -> npt.NDArray[np.bool_]:
    """Which cells hold a non-empty string (not None, not "").

    Inputs: names (object array). Output: bool array, same shape.
    Side effects: none. Exceptions: none.
    """
    return np.array([isinstance(name, str) and name != "" for name in names], np.bool_)


def _card_codes(pack_columns: Sequence[tuple[str, UUID]]) -> dict[UUID, int]:
    """A code per distinct card among pack_columns, in first-column order.

    Inputs: pack_columns. Output: dict card uuid -> code (0, 1, ...).
    Side effects: none. Exceptions: none.
    """
    result: dict[UUID, int] = {}
    for _, card_uuid in pack_columns:
        result.setdefault(card_uuid, len(result))
    return result
