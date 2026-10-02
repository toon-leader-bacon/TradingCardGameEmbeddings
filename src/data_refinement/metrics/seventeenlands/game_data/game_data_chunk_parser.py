"""GameDataChunkParser - the one place that knows a game_data CSV's
column names: built once per CSV from its header and the CardBinder,
then turns each pyarrow RecordBatch into a GameDataChunk.

Card-name matching is delegated to GameCardColumns.from_header()
(game_card_columns.py), so which header columns match which card is
identical to the row implementation. Each row's deck is identified by
build_chunk_decks() (chunk_decks.py). The scanner never sees the
binder; the driver builds this parser and hands it to scan_game_csv().
"""

from typing import Mapping, Sequence
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.chunk_decks import (
    build_chunk_decks,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameKeys,
    GameZone,
    ZoneCounts,
)
from src.schema.game_id import GameId

# Every scalar column a GameDataChunk carries, with its read type. A
# string column reads an empty cell as "" (pyarrow's default), never as
# null, so an unranked event's rank is "" (and an empty draft_id would
# pass through as "", as it did when the row metrics read it). Every
# game_data CSV under
# data/raw/17lands/ carries all seven (checked 2026-10-02), so a header
# missing one fails the whole CSV rather than one metric.
_SCALAR_TYPES: Mapping[str, pa.DataType] = {
    "won": pa.bool_(),
    "on_play": pa.bool_(),
    "num_turns": pa.int32(),
    "draft_id": pa.string(),
    "match_number": pa.int64(),
    "game_number": pa.int64(),
    "rank": pa.string(),
}


class GameDataChunkParser:
    """One CSV's column layout: matched card columns per zone, plus the
    scalar columns."""

    def __init__(
        self,
        header: tuple[str, ...],
        zone_columns: Mapping[GameZone, tuple[tuple[str, UUID], ...]],
        source_game: GameId,
    ) -> None:
        """Use from_header(); this constructor only stores parsed state.

        Inputs:
            header: the CSV's full column list, in file order.
            zone_columns: per zone, (header column, card uuid) for every
                matched column, in header order.
            source_game: stamped on every deck the chunks identify.
        Output: none (constructor). Side effects: none. Exceptions: none.
        """
        self._header = header
        self._zone_columns = zone_columns
        self._source_game = source_game

    @classmethod
    def from_header(
        cls,
        header: Sequence[str],
        card_binder: CardBinder,
        source_game: GameId,
    ) -> "GameDataChunkParser":
        """Match one CSV's header against card_binder (Factory Method).

        Inputs:
            header: the CSV's column names (e.g. pd.read_csv(path,
                nrows=0).columns).
            card_binder: populated for source_game; read only here.
            source_game: whose cards the column suffixes name.
        Output: a GameDataChunkParser.
        Side effects: none (no I/O).
        Exceptions: ValueError if a scalar column in _SCALAR_TYPES is
            missing from header.

        Example:
            >>> GameDataChunkParser.from_header(header, binder, GameId.MTG)
        """
        # Validate inputs: every scalar the chunk carries must exist
        _require_scalar_columns(header)

        # Reuse the row implementation's matching, so column -> card is
        # identical, then regroup its per-zone lists by GameZone
        game_columns = GameCardColumns.from_header(header, card_binder, source_game)
        zone_columns = _group_zone_columns(game_columns)

        return cls(tuple(header), zone_columns, source_game)

    def needed_columns(self) -> list[str]:
        """The columns the scanner must read: the matched card columns
        plus the scalars.

        Inputs: none. Output: list[str], header order.
        Side effects: none. Exceptions: none.

        Example:
            >>> pyarrow.csv.ConvertOptions(include_columns=parser.needed_columns())
        """
        # Card columns of every zone, then the scalars, in header order
        wanted = {
            column for columns in self._zone_columns.values() for column, _ in columns
        }
        wanted.update(_SCALAR_TYPES)
        return [column for column in self._header if column in wanted]

    def column_types(self) -> dict[str, pa.DataType]:
        """Read types for the scanner: every matched card column as a
        small integer, each scalar as _SCALAR_TYPES says.

        Inputs: none. Output: dict column -> pyarrow type.
        Side effects: none. Exceptions: none.

        Example:
            >>> pyarrow.csv.ConvertOptions(column_types=parser.column_types())
        """
        result: dict[str, pa.DataType] = dict(_SCALAR_TYPES)
        for columns in self._zone_columns.values():
            for column, _ in columns:
                result[column] = pa.int16()
        return result

    def parse(self, batch: pa.RecordBatch) -> GameDataChunk:
        """Turn one record batch into a GameDataChunk.

        Inputs: batch, read with needed_columns()/column_types().
        Output: GameDataChunk with every zone's ZoneCounts, the typed
            scalars, and each row's deck.
        Side effects: none.
        Exceptions: ValueError naming the column and first row offset
            if any scalar column holds a null.

        Example:
            >>> chunk = parser.parse(next(iter(reader)))
        """
        # Validate the scalars before building anything from them
        for column in _SCALAR_TYPES:
            _raise_on_null(batch, column)

        # Each zone: one count matrix over its matched columns
        zones = {
            zone: _read_zone_counts(batch, columns)
            for zone, columns in self._zone_columns.items()
        }
        keys = GameKeys(
            draft_id=_read_column(batch, "draft_id", np.object_),
            match_number=_read_column(batch, "match_number", np.int64),
            game_number=_read_column(batch, "game_number", np.int64),
        )

        return GameDataChunk(
            zones=zones,
            won=_read_column(batch, "won", np.bool_),
            on_play=_read_column(batch, "on_play", np.bool_),
            num_turns=_read_column(batch, "num_turns", np.int32),
            keys=keys,
            rank=_read_column(batch, "rank", np.object_),
            decks=build_chunk_decks(zones[GameZone.DECK], keys, self._source_game),
        )


def _require_scalar_columns(header: Sequence[str]) -> None:
    """Raise unless every _SCALAR_TYPES column is in header.

    Inputs: header. Output: none. Side effects: none.
    Exceptions: ValueError naming the missing columns.
    """
    missing = [column for column in _SCALAR_TYPES if column not in header]
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


def _read_column(
    batch: pa.RecordBatch, column: str, dtype: type[np.generic]
) -> npt.NDArray:
    """batch's column as a numpy array of dtype (nulls already
    rejected).

    Inputs: batch, column, dtype (e.g. np.bool_, np.int64, np.object_
        for strings). Output: shape (rows,).
    Side effects: none. Exceptions: none.
    """
    values = batch.column(column).to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype)
