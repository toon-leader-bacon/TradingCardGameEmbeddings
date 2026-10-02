"""GameDataChunkParser - the one place that knows a game_data CSV's
column names: built once per CSV from its header and the CardBinder,
then turns each pyarrow RecordBatch into a GameDataChunk.

Card-name matching is delegated to GameCardColumns.from_header()
(game_card_columns.py), so which header columns match which card is
identical to the row implementation. The scanner never sees the binder;
the driver builds this parser and hands it to scan_game_csv().
"""

from typing import Mapping, Sequence
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
    ZoneCounts,
)
from src.schema.game_id import GameId

# Scalar columns slice 1's metrics read. Later slices add rank etc.
_WON_COLUMN = "won"
_ON_PLAY_COLUMN = "on_play"
_NUM_TURNS_COLUMN = "num_turns"
_SCALAR_COLUMNS = (_WON_COLUMN, _ON_PLAY_COLUMN, _NUM_TURNS_COLUMN)


class GameDataChunkParser:
    """One CSV's column layout: matched card columns per zone, plus the
    scalar columns, and whether to keep a pandas copy for row metrics.
    """

    def __init__(
        self,
        header: tuple[str, ...],
        zone_columns: Mapping[GameZone, tuple[tuple[str, UUID], ...]],
        keep_source_frame: bool,
    ) -> None:
        """Use from_header(); this constructor only stores parsed state.

        Inputs:
            header: the CSV's full column list, in file order.
            zone_columns: per zone, (header column, card uuid) for every
                matched column, in header order.
            keep_source_frame: whether parse() also builds each chunk's
                source_frame (any row metric in the family).
        Output: none (constructor). Side effects: none. Exceptions: none.
        """
        self._header = header
        self._zone_columns = zone_columns
        self._keep_source_frame = keep_source_frame

    @classmethod
    def from_header(
        cls,
        header: Sequence[str],
        card_binder: CardBinder,
        source_game: GameId,
        keep_source_frame: bool,
    ) -> "GameDataChunkParser":
        """Match one CSV's header against card_binder (Factory Method).

        Inputs:
            header: the CSV's column names (e.g. pd.read_csv(path,
                nrows=0).columns).
            card_binder: populated for source_game; read only here.
            source_game: whose cards the column suffixes name.
            keep_source_frame: see __init__.
        Output: a GameDataChunkParser.
        Side effects: none (no I/O).
        Exceptions: ValueError if a scalar column in _SCALAR_COLUMNS is
            missing from header.

        Example:
            >>> GameDataChunkParser.from_header(header, binder, GameId.MTG,
            ...                                 keep_source_frame=True)
        """
        # Validate inputs: every scalar the chunk carries must exist
        _require_scalar_columns(header)

        # Reuse the row implementation's matching, so column -> card is
        # identical, then regroup its per-zone lists by GameZone
        game_columns = GameCardColumns.from_header(header, card_binder, source_game)
        zone_columns = _group_zone_columns(game_columns)

        return cls(tuple(header), zone_columns, keep_source_frame)

    def needed_columns(self) -> list[str]:
        """The columns the scanner must read: every column when a source
        frame is kept (row metrics may read any), else the matched card
        columns plus the scalars.

        Inputs: none. Output: list[str], header order.
        Side effects: none. Exceptions: none.

        Example:
            >>> pyarrow.csv.ConvertOptions(include_columns=parser.needed_columns())
        """
        if self._keep_source_frame:
            return list(self._header)

        # Card columns of every zone, then the scalars, in header order
        wanted = {
            column for columns in self._zone_columns.values() for column, _ in columns
        }
        wanted.update(_SCALAR_COLUMNS)
        return [column for column in self._header if column in wanted]

    def column_types(self) -> dict[str, pa.DataType]:
        """Read types for the scanner: every matched card column as a
        small integer, the scalars as bool/bool/int32. Other columns are
        left to pyarrow's inference.

        Inputs: none. Output: dict column -> pyarrow type.
        Side effects: none. Exceptions: none.

        Example:
            >>> pyarrow.csv.ConvertOptions(column_types=parser.column_types())
        """
        result: dict[str, pa.DataType] = {
            _WON_COLUMN: pa.bool_(),
            _ON_PLAY_COLUMN: pa.bool_(),
            _NUM_TURNS_COLUMN: pa.int32(),
        }
        for columns in self._zone_columns.values():
            for column, _ in columns:
                result[column] = pa.int16()
        return result

    def parse(self, batch: pa.RecordBatch) -> GameDataChunk:
        """Turn one record batch into a GameDataChunk.

        Inputs: batch, read with needed_columns()/column_types().
        Output: GameDataChunk with every zone's ZoneCounts, the typed
            scalars, and source_frame when keep_source_frame.
        Side effects: none.
        Exceptions: ValueError naming the column and first row offset
            if won, on_play or num_turns holds a null.

        Example:
            >>> chunk = parser.parse(next(iter(reader)))
        """
        # Validate the scalars before building anything from them
        for column in _SCALAR_COLUMNS:
            _raise_on_null(batch, column)

        # Each zone: one count matrix over its matched columns
        zones = {
            zone: _read_zone_counts(batch, columns)
            for zone, columns in self._zone_columns.items()
        }

        # Pandas copy only for a family still wrapping row metrics
        source_frame = batch.to_pandas() if self._keep_source_frame else None

        return GameDataChunk(
            zones=zones,
            won=_read_bool_column(batch, _WON_COLUMN),
            on_play=_read_bool_column(batch, _ON_PLAY_COLUMN),
            num_turns=_read_int32_column(batch, _NUM_TURNS_COLUMN),
            source_frame=source_frame,
        )


def _require_scalar_columns(header: Sequence[str]) -> None:
    """Raise unless every _SCALAR_COLUMNS entry is in header.

    Inputs: header. Output: none. Side effects: none.
    Exceptions: ValueError naming the missing columns.
    """
    missing = [column for column in _SCALAR_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"game_data CSV header lacks scalar columns {missing}")


def _group_zone_columns(
    game_columns: GameCardColumns,
) -> dict[GameZone, tuple[tuple[str, UUID], ...]]:
    """game_columns' five per-prefix column lists, keyed by GameZone.

    Inputs: game_columns. Output: dict with one entry per GameZone.
    Side effects: none. Exceptions: none.
    """
    per_zone = {
        GameZone.OPENING_HAND: game_columns.opening_hand_columns,
        GameZone.DRAWN: game_columns.drawn_columns,
        GameZone.TUTORED: game_columns.tutored_columns,
        GameZone.DECK: game_columns.deck_columns,
        GameZone.SIDEBOARD: game_columns.sideboard_columns,
    }
    return {zone: tuple(columns) for zone, columns in per_zone.items()}


def _read_zone_counts(
    batch: pa.RecordBatch, columns: tuple[tuple[str, UUID], ...]
) -> ZoneCounts:
    """One zone's (rows x columns) int16 count matrix; nulls become 0.

    Inputs: batch, columns (header column, card uuid) for this zone.
    Output: ZoneCounts (empty card_uuids and a (rows, 0) matrix when
        columns is empty).
    Side effects: none. Exceptions: none.
    """
    if not columns:
        return ZoneCounts(card_uuids=(), counts=np.zeros((batch.num_rows, 0), np.int16))

    # One numpy column per matched header column; a null cell means 0
    arrays = [
        batch.column(column).fill_null(0).to_numpy(zero_copy_only=False)
        for column, _ in columns
    ]
    counts = np.column_stack(arrays).astype(np.int16, copy=False)
    return ZoneCounts(card_uuids=tuple(uuid for _, uuid in columns), counts=counts)


def _raise_on_null(batch: pa.RecordBatch, column: str) -> None:
    """Raise if batch's column holds a null.

    Inputs: batch, column. Output: none. Side effects: none.
    Exceptions: ValueError naming column and the first null's offset.
    """
    values = batch.column(column)
    if values.null_count == 0:
        return
    first_null = int(np.flatnonzero(values.is_null().to_numpy(zero_copy_only=False))[0])
    raise ValueError(
        f"game_data column {column!r} is null at batch row {first_null} "
        f"({values.null_count} nulls in this batch)"
    )


def _read_bool_column(batch: pa.RecordBatch, column: str) -> npt.NDArray[np.bool_]:
    """batch's column as a numpy bool array (nulls already rejected).

    Inputs: batch, column. Output: shape (rows,).
    Side effects: none. Exceptions: none.
    """
    return np.asarray(
        batch.column(column).to_numpy(zero_copy_only=False), dtype=np.bool_
    )


def _read_int32_column(batch: pa.RecordBatch, column: str) -> npt.NDArray[np.int32]:
    """batch's column as a numpy int32 array (nulls already rejected).

    Inputs: batch, column. Output: shape (rows,).
    Side effects: none. Exceptions: none.
    """
    return np.asarray(
        batch.column(column).to_numpy(zero_copy_only=False), dtype=np.int32
    )
