"""ReplayDataChunk - one typed, numpy-backed block of 17lands replay_data
rows, the unit every replay_data metric's accumulate() receives (see
replay_data/README.md).

One row is one game. Besides its deck (deck_<name> counts) and keys, a
row has ~2,500 per-half-turn columns, f"{actor}_turn_{N}_{field}", each
holding a "|"-delimited list of Arena card ids. A chunk keeps every
ReplayField (the nine fields the metrics read; the other columns are
never read), each as a long-form TurnEvents table: one entry
per (row, actor, turn, card occurrence), in column order and, within a
cell, token order. Every metric is then a numpy group-by over these
entries; no metric ever splits a cell or visits a row in Python.

CARD CODES: every card a chunk names (deck columns and matched Arena
ids) has a small int code into the parser's append-only code table;
card_uuids is that table as of this chunk (a later chunk's table is a
superset). An Arena id that matched no card is kept with code UNMATCHED,
because a metric may need "did anything happen" (a kill) regardless of
whether the card is known.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from uuid import UUID

import numpy as np
import numpy.typing as npt

from src.data_refinement.seventeenlands.chunk_decks import (
    ChunkDecks,
    GameKeys,
)
from src.data_refinement.seventeenlands.zone_counts import ZoneCounts

# The code of an Arena id that matched no card
UNMATCHED = -1


class Actor(IntEnum):
    """Whose half-turn: the 17lands user or their opponent. The value is
    the actor's index in a half-turn id; label is the column prefix."""

    USER = 0
    OPPO = 1

    @property
    def label(self) -> str:
        """The column prefix: "user" or "oppo".

        Inputs: none. Output: str. Side effects: none. Exceptions: none.
        """
        return self.name.lower()


class ReplayField(Enum):
    """A per-half-turn card-list field a replay metric reads, valued by
    its column suffix (f"{actor}_turn_{N}_{value}")."""

    CREATURES_CAST = "creatures_cast"
    NON_CREATURES_CAST = "non_creatures_cast"
    CREATURES_ATTACKED = "creatures_attacked"
    CREATURES_BLOCKING = "creatures_blocking"
    CREATURES_UNBLOCKED = "creatures_unblocked"
    USER_CREATURES_KILLED_COMBAT = "user_creatures_killed_combat"
    OPPO_CREATURES_KILLED_COMBAT = "oppo_creatures_killed_combat"
    CARDS_DISCARDED = "cards_discarded"
    CARDS_TUTORED = "cards_tutored"


# The fields a card is cast in
CAST_FIELDS = (ReplayField.CREATURES_CAST, ReplayField.NON_CREATURES_CAST)


@dataclass(frozen=True)
class TurnEvents:
    """One field's card occurrences over a chunk, long form.

    rows: each entry's row in the chunk (int32).
    actors: each entry's Actor value (int8).
    turns: each entry's turn number, the actor's own counter (int16).
    codes: each entry's card code, UNMATCHED for an unknown Arena id
        (int32).
    turn_span: one more than the largest turn number any column of the
        CSV names; half_turn_ids() encodes with it.
    The four arrays have one entry per occurrence, so a card listed
    twice in a cell (two copies) is two entries.
    """

    rows: npt.NDArray[np.int32]
    actors: npt.NDArray[np.int8]
    turns: npt.NDArray[np.int16]
    codes: npt.NDArray[np.int32]
    turn_span: int

    def __post_init__(self) -> None:
        """Reject arrays of different lengths.

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError naming the lengths.
        """
        lengths = {
            "rows": self.rows.shape[0],
            "actors": self.actors.shape[0],
            "turns": self.turns.shape[0],
            "codes": self.codes.shape[0],
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"TurnEvents arrays differ in length: {lengths}")

    def __len__(self) -> int:
        """The entry count.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return int(self.codes.shape[0])

    def select(self, keep: npt.NDArray[np.bool_]) -> "TurnEvents":
        """The entries where keep is True, in order.

        Inputs: keep (bool, one per entry). Output: TurnEvents.
        Side effects: none. Exceptions: none.

        Example:
            >>> attacked.select(attacked.codes != UNMATCHED)
        """
        return TurnEvents(
            self.rows[keep],
            self.actors[keep],
            self.turns[keep],
            self.codes[keep],
            self.turn_span,
        )

    def matched(self) -> "TurnEvents":
        """The entries whose card is known (code != UNMATCHED).

        Inputs: none. Output: TurnEvents.
        Side effects: none. Exceptions: none.

        Example:
            >>> chunk.events[ReplayField.CREATURES_CAST].matched()
        """
        return self.select(self.codes != UNMATCHED)

    def for_actor(self, actor: Actor) -> "TurnEvents":
        """The entries of one actor's half-turns.

        Inputs: actor. Output: TurnEvents.
        Side effects: none. Exceptions: none.
        """
        return self.select(self.actors == actor.value)

    def half_turn_ids(self) -> npt.NDArray[np.int64]:
        """Each entry's half-turn as one int, (row * 2 + actor) *
        turn_span + turn: equal for entries of one half-turn, and sorting
        by it orders half-turns by row, then user before oppo, then turn.
        split_half_turn_ids() is the inverse.

        Inputs: none.
        Output: int64 array, one per entry.
        Side effects: none. Exceptions: none.

        Example:
            >>> attacked.half_turn_ids()
        """
        return (
            self.rows.astype(np.int64) * 2 + self.actors
        ) * self.turn_span + self.turns

    def split_half_turn_ids(
        self, half_turn_ids: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """The inverse of half_turn_ids(), under this table's turn_span.

        Inputs: half_turn_ids (any ids half_turn_ids() could produce).
        Output: (rows, actors, turns), int64 arrays like half_turn_ids.
        Side effects: none. Exceptions: none.

        Example:
            >>> rows, actors, turns = attacked.split_half_turn_ids(ids)
        """
        row_actor, turns = np.divmod(half_turn_ids, self.turn_span)
        rows, actors = np.divmod(row_actor, 2)
        return rows, actors, turns

    def rows_naming(
        self, codes: npt.NDArray[np.int32], row_count: int
    ) -> npt.NDArray[np.bool_]:
        """Whether each card in codes appears among these entries on each
        row: lines the entries up against any card list (e.g. "was this
        deck column's card cast by the user this game?").

        Inputs: codes (card codes, shape (n,); UNMATCHED never matches),
            row_count (the chunk's row count).
        Output: bool array, shape (row_count, n).
        Side effects: none. Exceptions: none.

        Example:
            >>> user_cast.rows_naming(chunk.deck_codes, len(chunk))
        """
        result = np.zeros((row_count, codes.shape[0]), np.bool_)

        # The entries naming a listed card, and each one's listed positions
        # (a card listed twice, e.g. two deck columns, has two)
        listed = np.isin(self.codes, codes) & (self.codes != UNMATCHED)
        entry_rows, entry_codes = self.rows[listed], self.codes[listed]
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        first = np.searchsorted(sorted_codes, entry_codes, side="left")
        spans = np.searchsorted(sorted_codes, entry_codes, side="right") - first

        # Expand each entry to one (row, position) pair per listed position
        pair_rows = np.repeat(entry_rows, spans)
        offsets = np.arange(pair_rows.shape[0]) - np.repeat(
            np.cumsum(spans) - spans, spans
        )
        result[pair_rows, order[np.repeat(first, spans) + offsets]] = True
        return result

    @staticmethod
    def combine(*parts: "TurnEvents") -> "TurnEvents":
        """Several fields' entries as one table (e.g. creatures cast and
        non-creatures cast), parts in order.

        Inputs: parts (one or more TurnEvents of one CSV).
        Output: TurnEvents.
        Side effects: none.
        Exceptions: ValueError if parts is empty or their turn_spans
            differ.

        Example:
            >>> TurnEvents.combine(creatures_cast, non_creatures_cast)
        """
        if not parts:
            raise ValueError("TurnEvents.combine needs at least one part")
        turn_spans = {part.turn_span for part in parts}
        if len(turn_spans) != 1:
            raise ValueError(f"TurnEvents.combine: turn_spans differ: {turn_spans}")
        if len(parts) == 1:
            return parts[0]
        return TurnEvents(
            np.concatenate([part.rows for part in parts]),
            np.concatenate([part.actors for part in parts]),
            np.concatenate([part.turns for part in parts]),
            np.concatenate([part.codes for part in parts]),
            parts[0].turn_span,
        )


@dataclass(frozen=True)
class ReplayDataChunk:
    """One record batch of replay_data rows (one game each).

    deck: deck_<name> counts, one column per matched header column.
    deck_codes: each deck column's card code, shape (deck columns,).
    events: one TurnEvents per ReplayField (every field, read-only
        mapping; a field the CSV lacks is an empty table).
    card_uuids: the code table as of this chunk: card_uuids[code].
    keys: each row's (draft_id, match_number, game_number).
    num_turns: each row's num_turns (int32).
    decks: each row's deck, identified once (ChunkDecks).
    """

    deck: ZoneCounts
    deck_codes: npt.NDArray[np.int32]
    events: Mapping[ReplayField, TurnEvents]
    card_uuids: tuple[UUID, ...]
    keys: GameKeys
    num_turns: npt.NDArray[np.int32]
    decks: ChunkDecks

    def __post_init__(self) -> None:
        """Reject a chunk whose per-row fields disagree on row count, a
        missing ReplayField, an event row out of range, a code outside
        card_uuids, or deck_codes that don't name deck.card_uuids
        (card_uuids[deck_codes[i]] must be deck.card_uuids[i]).

        Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError naming the field.
        """
        rows = len(self)
        per_row = {
            "deck.counts": self.deck.counts.shape[0],
            "keys.draft_id": self.keys.draft_id.shape[0],
            "keys.match_number": self.keys.match_number.shape[0],
            "keys.game_number": self.keys.game_number.shape[0],
            "decks.row_deck": self.decks.row_deck.shape[0],
        }
        for name, length in per_row.items():
            if length != rows:
                raise ValueError(
                    f"ReplayDataChunk.{name} has {length} rows, not {rows}"
                )

        missing = [field.name for field in ReplayField if field not in self.events]
        if missing:
            raise ValueError(f"ReplayDataChunk.events is missing fields {missing}")
        for field, events in self.events.items():
            self._check_events(field, events)

        named = tuple(self.card_uuids[code] for code in self.deck_codes)
        if named != self.deck.card_uuids:
            raise ValueError(
                "ReplayDataChunk.deck_codes don't name deck.card_uuids in order"
            )

    def _check_events(self, field: ReplayField, events: TurnEvents) -> None:
        """Reject one field's entries if a row is outside the chunk or a
        code outside card_uuids (UNMATCHED aside).

        Inputs: field (named in the error), events.
        Output: none. Side effects: none.
        Exceptions: ValueError naming the field.
        """
        if events.rows.size and (
            events.rows.min() < 0 or events.rows.max() >= len(self)
        ):
            raise ValueError(f"ReplayDataChunk.events[{field.name}] row out of range")
        if events.codes.size and (
            events.codes.min() < UNMATCHED or events.codes.max() >= len(self.card_uuids)
        ):
            raise ValueError(
                f"ReplayDataChunk.events[{field.name}] code outside card_uuids"
            )

    def __len__(self) -> int:
        """The chunk's row count.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return int(self.num_turns.shape[0])

    def events_for(self, fields: tuple[ReplayField, ...]) -> TurnEvents:
        """Several fields' entries as one table, fields in order.

        Inputs: fields (one or more ReplayFields).
        Output: TurnEvents (matched and unmatched entries).
        Side effects: none.
        Exceptions: ValueError if fields is empty.

        Example:
            >>> chunk.events_for(CAST_FIELDS).matched()
        """
        return TurnEvents.combine(*(self.events[field] for field in fields))
