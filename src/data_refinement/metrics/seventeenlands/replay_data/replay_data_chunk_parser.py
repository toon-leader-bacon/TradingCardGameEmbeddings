"""ReplayDataChunkParser - one replay_data CSV's column layout, turning
each pyarrow record batch into one ReplayDataChunk (see
replay_data/README.md). A ChunkParser (../chunk_scanner.py).

Deck matching (deck_<name> suffixes) and Arena-id matching are
ReplayCardColumns' (replay_card_columns.py), set up once per CSV. Every
per-half-turn field column is read as a string; a cell's "|"-delimited
tokens are split and flattened with pyarrow compute, and each distinct
token is normalized to the Arena alias ledger's form
(str(int(float(token))), so "104936" and "104936.0" match alike) and
matched once.

The f"{actor}_turn_{N}_{field}" column grammar lives here alone
(_field_columns), over Actor.label and ReplayField.value.
"""

import re
from collections.abc import Sequence
from types import MappingProxyType
from typing import NamedTuple
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.batch_columns import (
    CARD_COUNT_TYPE,
    raise_on_null,
    read_column,
    read_zone_counts,
)
from src.data_refinement.metrics.seventeenlands.chunk_scanner import (
    UnsupportedCsvLayout,
)
from src.data_refinement.metrics.seventeenlands.chunk_decks import (
    GameKeys,
    build_chunk_decks,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    UNMATCHED,
    Actor,
    ReplayDataChunk,
    ReplayField,
    TurnEvents,
)
from src.schema.game_id import GameId

# Every scalar column a ReplayDataChunk is built from, with its read type
_SCALAR_TYPES: dict[str, pa.DataType] = {
    "draft_id": pa.string(),
    "match_number": pa.int64(),
    "game_number": pa.int64(),
    "num_turns": pa.int32(),
}
# Only the oldest exports (AFR, STX) have this column (today's is
# game_number). That layout also has no deck_ columns and no
# match_number, so neither the deck metrics nor GameKeys can be built.
_OLD_LAYOUT_COLUMN = "game_index"
_DECK_FAMILY_LABEL = "replay_data"
# One per-half-turn field column: f"{actor}_turn_{N}_{field}"
_FIELD_COLUMN_PATTERN = re.compile(
    r"^(?P<actor>user|oppo)_turn_(?P<turn>\d+)_(?P<field>.+)$"
)


class FieldColumn(NamedTuple):
    """One per-half-turn column: its name, actor and turn number."""

    name: str
    actor: Actor
    turn: int


class CardCodeTable:
    """An append-only card uuid -> int code table: codes are stable for a
    CSV, so metrics can tally by code across chunks."""

    def __init__(self, card_uuids: Sequence[UUID]) -> None:
        """Start with card_uuids coded 0, 1, ... (duplicates share one
        code).

        Inputs: card_uuids (seed cards, e.g. the deck columns').
        Output: none (constructor). Side effects: none. Exceptions: none.
        """
        self._codes: dict[UUID, int] = {}
        self._uuids: list[UUID] = []
        for card_uuid in card_uuids:
            self.code_for(card_uuid)

    def code_for(self, card_uuid: UUID) -> int:
        """card_uuid's code, adding it if new.

        Inputs: card_uuid. Output: int code.
        Side effects: may append card_uuid to the table.
        Exceptions: none.

        Example:
            >>> table.code_for(owlbear_uuid)
            0
        """
        code = self._codes.get(card_uuid)
        if code is None:
            code = len(self._uuids)
            self._codes[card_uuid] = code
            self._uuids.append(card_uuid)
        return code

    def snapshot(self) -> tuple[UUID, ...]:
        """The table as it stands: snapshot()[code] is that code's uuid.

        Inputs: none. Output: tuple of uuids.
        Side effects: none. Exceptions: none.
        """
        return tuple(self._uuids)


class ReplayDataChunkParser:
    """One CSV's column layout: matched deck columns, the field columns
    per ReplayField, and the scalar columns."""

    def __init__(
        self,
        header: tuple[str, ...],
        replay_columns: ReplayCardColumns,
        field_columns: dict[ReplayField, list[FieldColumn]],
        source_game: GameId,
    ) -> None:
        """Use from_header(); this constructor only stores parsed state
        and seeds the code table with the deck columns' cards.

        Inputs:
            header: the CSV's full column list, in file order.
            replay_columns: the header's matched deck columns, and the
                Arena-id cache tokens are matched through.
            field_columns: per field, every (column, actor, turn) the
                header has, in header order.
            source_game: stamped on every deck the chunks identify.
        Output: none (constructor). Side effects: none. Exceptions: none.
        """
        self._header = header
        self._replay_columns = replay_columns
        self._field_columns = field_columns
        self._source_game = source_game
        self._code_table = CardCodeTable(
            [card_uuid for _, card_uuid in replay_columns.deck_columns]
        )
        self._deck_codes = np.array(
            [self._code_table.code_for(u) for _, u in replay_columns.deck_columns],
            np.int32,
        )
        self._turn_span = 1 + max(
            (column.turn for columns in field_columns.values() for column in columns),
            default=0,
        )
        self._column_types = self._read_types()
        self._token_codes: dict[str, int] = {}

    @classmethod
    def from_header(
        cls, header: Sequence[str], card_binder: CardBinder, source_game: GameId
    ) -> "ReplayDataChunkParser":
        """Match one CSV's header against card_binder (Factory Method).

        Inputs: header (the CSV's column names), card_binder (populated
            for source_game, with Arena aliases), source_game.
        Output: a ReplayDataChunkParser.
        Side effects: none (no I/O).
        Exceptions: UnsupportedCsvLayout for the AFR/STX layout (no deck
            columns, no match_number); otherwise ValueError if a column
            in _SCALAR_TYPES is missing.

        Example:
            >>> ReplayDataChunkParser.from_header(header, binder, GameId.MTG)
        """
        if _OLD_LAYOUT_COLUMN in header:
            raise UnsupportedCsvLayout(
                "replay_data CSV has the AFR/STX layout (game_index, no deck "
                "columns, no match_number), which the metrics cannot read"
            )
        missing = [column for column in _SCALAR_TYPES if column not in header]
        if missing:
            raise ValueError(f"replay_data CSV header lacks scalar columns {missing}")
        replay_columns = ReplayCardColumns.from_header(header, card_binder, source_game)
        return cls(tuple(header), replay_columns, _field_columns(header), source_game)

    def needed_columns(self) -> list[str]:
        """The columns the scanner must read: matched deck columns, every
        field column, and the scalars.

        Inputs: none. Output: list[str], header order.
        Side effects: none. Exceptions: none.

        Example:
            >>> pyarrow.csv.ConvertOptions(include_columns=parser.needed_columns())
        """
        return [column for column in self._header if column in self._column_types]

    def column_types(self) -> dict[str, pa.DataType]:
        """Read types: deck columns as float32, field columns as strings
        (a single-id cell would otherwise read as a number), scalars as
        _SCALAR_TYPES says.

        Inputs: none. Output: dict column -> pyarrow type.
        Side effects: none. Exceptions: none.

        Example:
            >>> pyarrow.csv.ConvertOptions(column_types=parser.column_types())
        """
        return dict(self._column_types)

    def _read_types(self) -> dict[str, pa.DataType]:
        """Every needed column's read type, built once per CSV.

        Inputs: none. Output: dict column -> pyarrow type.
        Side effects: none. Exceptions: none.
        """
        result: dict[str, pa.DataType] = dict(_SCALAR_TYPES)
        for deck_column, _ in self._replay_columns.deck_columns:
            result[deck_column] = CARD_COUNT_TYPE
        for columns in self._field_columns.values():
            for field_column in columns:
                result[field_column.name] = pa.string()
        return result

    def parse(self, batch: pa.RecordBatch) -> ReplayDataChunk:
        """Turn one record batch into a ReplayDataChunk.

        Inputs: batch, read with needed_columns()/column_types().
        Output: ReplayDataChunk.
        Side effects: may add cards to the code table and Arena ids to
            the ReplayCardColumns cache.
        Exceptions: ValueError naming the column and first row offset if
            a scalar column holds a null.

        Example:
            >>> chunk = parser.parse(next(iter(reader)))
        """
        # Validate the scalars before building anything from them
        for column in _SCALAR_TYPES:
            raise_on_null(batch, column)

        # The deck, the per-field events, the keys and each row's deck
        deck = read_zone_counts(batch, self._replay_columns.deck_columns)
        events = MappingProxyType(
            {
                field: self._read_turn_events(batch, columns)
                for field, columns in self._field_columns.items()
            }
        )
        keys = GameKeys(
            draft_id=read_column(batch, "draft_id", np.object_),
            match_number=read_column(batch, "match_number", np.int64),
            game_number=read_column(batch, "game_number", np.int64),
        )
        return ReplayDataChunk(
            deck=deck,
            deck_codes=self._deck_codes,
            events=events,
            card_uuids=self._code_table.snapshot(),
            keys=keys,
            num_turns=read_column(batch, "num_turns", np.int32),
            decks=build_chunk_decks(deck, keys, self._source_game, _DECK_FAMILY_LABEL),
        )

    def _read_turn_events(
        self, batch: pa.RecordBatch, columns: list[FieldColumn]
    ) -> TurnEvents:
        """One field's entries: every column's cells split on "|" and
        flattened (pyarrow compute), tagged with the column's actor and
        turn, tokens coded.

        Inputs: batch, columns (the field's (column, actor, turn)s).
        Output: TurnEvents, columns in order, tokens in cell order.
        Side effects: as _codes_for_tokens.
        Exceptions: ValueError if a token is not a number.
        """
        token_parts: list[pa.Array] = []
        row_parts: list[npt.NDArray[np.int32]] = []
        actor_parts: list[npt.NDArray[np.int8]] = []
        turn_parts: list[npt.NDArray[np.int16]] = []

        # Split each column's cells into tokens, tagging each token
        for column in columns:
            cells = pc.split_pattern(batch.column(column.name), "|")
            lengths = pc.fill_null(pc.list_value_length(cells), 0).to_numpy()
            column_rows = np.repeat(np.arange(len(cells), dtype=np.int32), lengths)
            token_parts.append(pc.list_flatten(cells))
            row_parts.append(column_rows)
            actor_parts.append(np.full(column_rows.shape, column.actor, np.int8))
            turn_parts.append(np.full(column_rows.shape, column.turn, np.int16))

        if not token_parts:
            return _empty_turn_events(self._turn_span)

        # An empty token (an empty cell) names no card
        tokens = pa.concat_arrays(token_parts)
        named = pc.not_equal(tokens, "").to_numpy(zero_copy_only=False)
        return TurnEvents(
            rows=np.concatenate(row_parts)[named],
            actors=np.concatenate(actor_parts)[named],
            turns=np.concatenate(turn_parts)[named],
            codes=self._codes_for_tokens(tokens.filter(pa.array(named))),
            turn_span=self._turn_span,
        )

    def _codes_for_tokens(self, tokens: pa.Array) -> npt.NDArray[np.int32]:
        """Raw Arena-id tokens as card codes: each distinct token is
        normalized and matched once (cached across chunks); an unmatched
        one codes as UNMATCHED.

        Inputs: tokens (pyarrow string array, no nulls or empties).
        Output: int32 codes, one per token.
        Side effects: may add cards to the code table and Arena ids to
            the ReplayCardColumns cache.
        Exceptions: ValueError if a token is not a number.
        """
        encoded = pc.dictionary_encode(tokens)
        distinct_codes = np.array(
            [self._code_for_token(token) for token in encoded.dictionary.to_pylist()],
            np.int32,
        )
        return distinct_codes[encoded.indices.to_numpy(zero_copy_only=False)]

    def _code_for_token(self, token: str) -> int:
        """One raw Arena-id token's card code (cached across chunks),
        UNMATCHED if no card has that Arena id.

        Inputs: token (e.g. "104936" or "104936.0").
        Output: int code.
        Side effects: may add a card to the code table and the Arena id
            to the ReplayCardColumns cache.
        Exceptions: ValueError if token is not a number.
        """
        code = self._token_codes.get(token)
        if code is None:
            card_uuid = self._replay_columns.uuid_for_arena_id(str(int(float(token))))
            code = (
                UNMATCHED if card_uuid is None else self._code_table.code_for(card_uuid)
            )
            self._token_codes[token] = code
        return code


def _field_columns(header: Sequence[str]) -> dict[ReplayField, list[FieldColumn]]:
    """Per ReplayField, every f"{actor}_turn_{N}_{field}" column header
    has, as FieldColumns in header order. The column's suffix must equal
    ReplayField.value exactly (user_creatures_killed_combat is its own
    field, never a match for creatures_killed_combat), and actor is
    Actor.label.

    Inputs: header. Output: dict with one (possibly empty) list per
        ReplayField.
    Side effects: none. Exceptions: none.
    """
    result: dict[ReplayField, list[FieldColumn]] = {field: [] for field in ReplayField}
    fields = {field.value: field for field in ReplayField}
    actors = {actor.label: actor for actor in Actor}
    for column in header:
        match = _FIELD_COLUMN_PATTERN.match(column)
        if match is None or match["field"] not in fields:
            continue
        result[fields[match["field"]]].append(
            FieldColumn(column, actors[match["actor"]], int(match["turn"]))
        )
    return result


def _empty_turn_events(turn_span: int) -> TurnEvents:
    """A field's entries when the CSV has none of its columns.

    Inputs: turn_span. Output: TurnEvents with no entries.
    Side effects: none. Exceptions: none.
    """
    return TurnEvents(
        rows=np.empty(0, np.int32),
        actors=np.empty(0, np.int8),
        turns=np.empty(0, np.int16),
        codes=np.empty(0, np.int32),
        turn_span=turn_span,
    )
