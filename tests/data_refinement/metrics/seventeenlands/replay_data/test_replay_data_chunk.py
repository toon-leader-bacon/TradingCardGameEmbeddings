"""Tests for replay_data_chunk.py: TurnEvents' long-form operations and
ReplayDataChunk's construction invariants."""

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    UNMATCHED,
    Actor,
    ReplayField,
    TurnEvents,
)
from tests.data_refinement.metrics.seventeenlands.replay_data._replay_fixtures import (
    binder_with_cards,
    parse_rows,
    row,
)


def _events(
    rows: list[int],
    codes: list[int],
    actors: list[int] | None = None,
    turns: list[int] | None = None,
    turn_span: int = 5,
) -> TurnEvents:
    count = len(rows)
    return TurnEvents(
        rows=np.array(rows, np.int32),
        actors=np.array(actors if actors is not None else [0] * count, np.int8),
        turns=np.array(turns if turns is not None else [1] * count, np.int16),
        codes=np.array(codes, np.int32),
        turn_span=turn_span,
    )


class TestTurnEvents:
    def test_arrays_of_different_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            TurnEvents(
                rows=np.zeros(2, np.int32),
                actors=np.zeros(1, np.int8),
                turns=np.zeros(2, np.int16),
                codes=np.zeros(2, np.int32),
                turn_span=3,
            )

    def test_matched_drops_unmatched_entries(self) -> None:
        events = _events([0, 0, 1], [3, UNMATCHED, 4])

        assert events.matched().codes.tolist() == [3, 4]

    def test_for_actor_keeps_one_actors_entries(self) -> None:
        events = _events([0, 0], [3, 4], actors=[Actor.USER, Actor.OPPO])

        assert events.for_actor(Actor.OPPO).codes.tolist() == [4]

    def test_half_turn_ids_round_trip(self) -> None:
        events = _events([0, 2, 2], [1, 1, 1], actors=[1, 0, 1], turns=[4, 0, 3])

        rows, actors, turns = events.split_half_turn_ids(events.half_turn_ids())

        assert rows.tolist() == [0, 2, 2]
        assert actors.tolist() == [1, 0, 1]
        assert turns.tolist() == [4, 0, 3]

    def test_half_turn_ids_sort_by_row_then_user_first_then_turn(self) -> None:
        events = _events(
            [1, 0, 0, 0], [1, 1, 1, 1], actors=[0, 1, 0, 0], turns=[1, 1, 2, 1]
        )

        order = np.argsort(events.half_turn_ids()).tolist()

        assert order == [3, 2, 1, 0]

    def test_rows_naming_lines_entries_up_against_a_card_list(self) -> None:
        events = _events([0, 1, 1], [7, 8, UNMATCHED])

        named = events.rows_naming(np.array([8, 7, 9], np.int32), row_count=3)

        assert named.tolist() == [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]

    def test_rows_naming_marks_every_position_of_a_card_listed_twice(self) -> None:
        events = _events([0], [7])

        named = events.rows_naming(np.array([7, 5, 7], np.int32), row_count=1)

        assert named.tolist() == [[True, False, True]]

    def test_unmatched_never_names_a_listed_unmatched_code(self) -> None:
        events = _events([0], [UNMATCHED])

        named = events.rows_naming(np.array([UNMATCHED], np.int32), row_count=1)

        assert named.tolist() == [[False]]

    def test_combine_concatenates_parts_in_order(self) -> None:
        combined = TurnEvents.combine(_events([0], [1]), _events([1], [2]))

        assert combined.codes.tolist() == [1, 2]
        assert combined.rows.tolist() == [0, 1]

    def test_combine_needs_a_part(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            TurnEvents.combine()

    def test_combine_rejects_differing_turn_spans(self) -> None:
        with pytest.raises(ValueError, match="turn_spans differ"):
            TurnEvents.combine(_events([0], [1]), _events([0], [1], turn_span=9))


class TestReplayDataChunk:
    def test_a_parsed_chunk_has_every_field(self, tmp_path: Path) -> None:
        chunk = parse_rows(tmp_path, binder_with_cards(), [row()])

        assert set(chunk.events) == set(ReplayField)
        assert len(chunk) == 1

    def test_a_missing_field_is_rejected(self, tmp_path: Path) -> None:
        chunk = parse_rows(tmp_path, binder_with_cards(), [row()])
        events = dict(chunk.events)
        del events[ReplayField.CARDS_TUTORED]

        with pytest.raises(ValueError, match="CARDS_TUTORED"):
            replace(chunk, events=MappingProxyType(events))

    def test_an_event_row_outside_the_chunk_is_rejected(self, tmp_path: Path) -> None:
        chunk = parse_rows(tmp_path, binder_with_cards(), [row()])
        events = dict(chunk.events)
        events[ReplayField.CREATURES_CAST] = _events([5], [0], turn_span=3)

        with pytest.raises(ValueError, match="row out of range"):
            replace(chunk, events=MappingProxyType(events))

    def test_a_code_outside_the_code_table_is_rejected(self, tmp_path: Path) -> None:
        chunk = parse_rows(tmp_path, binder_with_cards(), [row()])
        events = dict(chunk.events)
        events[ReplayField.CREATURES_CAST] = _events([0], [len(chunk.card_uuids)])

        with pytest.raises(ValueError, match="code outside"):
            replace(chunk, events=MappingProxyType(events))

    def test_a_row_count_mismatch_is_rejected(self, tmp_path: Path) -> None:
        chunk = parse_rows(tmp_path, binder_with_cards(), [row()])

        with pytest.raises(ValueError, match="num_turns|rows"):
            replace(chunk, num_turns=np.zeros(2, np.int32))

    def test_deck_codes_must_name_the_deck_columns(self, tmp_path: Path) -> None:
        chunk = parse_rows(tmp_path, binder_with_cards(), [row()])

        with pytest.raises(ValueError, match="deck_codes"):
            replace(chunk, deck_codes=chunk.deck_codes[::-1].copy())
