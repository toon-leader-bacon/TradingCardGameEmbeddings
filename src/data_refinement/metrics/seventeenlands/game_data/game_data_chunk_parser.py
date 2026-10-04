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
from src.data_refinement.metrics.seventeenlands.chunk_decks import (
    build_chunk_decks,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_refinement.metrics.seventeenlands.chunk_decks import (
    GameKeys,
)
from src.data_refinement.metrics.seventeenlands.batch_columns import (
    CARD_COUNT_TYPE,
    raise_on_null,
    read_column,
    read_zone_counts,
)
from src.schema.game_id import GameId

# Every scalar column a GameDataChunk carries, with its read type. A
# string column reads an empty cell as "" (pyarrow's default), never as
# null, so an unranked event's rank is "" (and an empty draft_id would
# pass through as "", as it did when the row metrics read it). A header
# missing a required scalar fails the whole CSV rather than one metric;
# an optional one (_OPTIONAL_SCALAR_DEFAULTS) reads as its default.
_SCALAR_TYPES: Mapping[str, pa.DataType] = {
    "won": pa.bool_(),
    "on_play": pa.bool_(),
    "num_turns": pa.int32(),
    "draft_id": pa.string(),
    "match_number": pa.int64(),
    "game_number": pa.int64(),
    "rank": pa.string(),
}

# Optional scalars and the value every row gets when the CSV lacks the
# column. The older exports (AFR, KHM, MID, STX, VOW; 10 of 132 CSVs)
# have no match_number and restart game_number at 1 for each match;
# match_number reads as 0 there, the deck box extraction stage's
# convention (deck_box/seventeenlands_game_data/extraction_stage.py).
# Modern exports number matches from 1, so 0 always means "unknown".
_OPTIONAL_SCALAR_DEFAULTS: Mapping[str, int] = {"match_number": 0}


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
        Exceptions: ValueError if a required scalar column (in
            _SCALAR_TYPES, not _OPTIONAL_SCALAR_DEFAULTS) is missing from
            header.

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
        result: dict[str, pa.DataType] = {
            column: kind
            for column, kind in _SCALAR_TYPES.items()
            if column in self._header
        }
        # float32, not int16: some exports (e.g. BRO.PremierDraft) write
        # counts as "0.0"; read_zone_counts() narrows them to int16
        for columns in self._zone_columns.values():
            for column, _ in columns:
                result[column] = CARD_COUNT_TYPE
        return result

    def parse(self, batch: pa.RecordBatch) -> GameDataChunk:
        """Turn one record batch into a GameDataChunk.

        Inputs: batch, read with needed_columns()/column_types().
        Output: GameDataChunk with every zone's ZoneCounts, the typed
            scalars, and each row's deck.
        Side effects: none.
        Exceptions: ValueError naming the column and first row offset
            if any scalar column holds a null. A missing optional scalar
            (match_number) reads as its default instead.

        Example:
            >>> chunk = parser.parse(next(iter(reader)))
        """
        # Validate the scalars before building anything from them
        for column in _SCALAR_TYPES:
            if column in batch.schema.names:
                raise_on_null(batch, column)

        # Each zone: one count matrix over its matched columns
        zones = {
            zone: read_zone_counts(batch, columns)
            for zone, columns in self._zone_columns.items()
        }
        keys = GameKeys(
            draft_id=read_column(batch, "draft_id", np.object_),
            match_number=_read_optional_column(batch, "match_number"),
            game_number=read_column(batch, "game_number", np.int64),
        )

        return GameDataChunk(
            zones=zones,
            won=read_column(batch, "won", np.bool_),
            on_play=read_column(batch, "on_play", np.bool_),
            num_turns=read_column(batch, "num_turns", np.int32),
            keys=keys,
            rank=read_column(batch, "rank", np.object_),
            decks=build_chunk_decks(
                zones[GameZone.DECK], keys, self._source_game, "game_data"
            ),
        )


def _require_scalar_columns(header: Sequence[str]) -> None:
    """Raise unless every _SCALAR_TYPES column is in header.

    Inputs: header. Output: none. Side effects: none.
    Exceptions: ValueError naming the missing columns.
    """
    missing = [
        column
        for column in _SCALAR_TYPES
        if column not in header and column not in _OPTIONAL_SCALAR_DEFAULTS
    ]
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


def _read_optional_column(batch: pa.RecordBatch, column: str) -> npt.NDArray:
    """An optional int64 scalar column, or its default on every row when
    the CSV lacks it.

    Inputs: batch, column (a key of _OPTIONAL_SCALAR_DEFAULTS).
    Output: int64 array, shape (batch.num_rows,).
    Side effects: none. Exceptions: none.
    """
    if column in batch.schema.names:
        return read_column(batch, column, np.int64)
    return np.full(batch.num_rows, _OPTIONAL_SCALAR_DEFAULTS[column], np.int64)
