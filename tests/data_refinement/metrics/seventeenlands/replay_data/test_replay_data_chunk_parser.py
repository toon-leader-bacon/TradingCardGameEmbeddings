"""Tests for replay_data_chunk_parser.py: the field-column grammar, token
splitting and coding, and the append-only code table."""

from pathlib import Path

import pytest

from src.data_refinement.metrics.seventeenlands.chunk_scanner import (
    UnsupportedCsvLayout,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    UNMATCHED,
    Actor,
    ReplayField,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk_parser import (
    CardCodeTable,
    FieldColumn,
    ReplayDataChunkParser,
)
from src.schema.game_id import GameId
from tests.data_refinement.metrics.seventeenlands.replay_data._replay_fixtures import (
    HEADER,
    MORNINGSTAR,
    MORNINGSTAR_ID,
    OWLBEAR,
    OWLBEAR_ID,
    UNKNOWN_ID,
    binder_with_cards,
    field_column,
    parse_rows,
    parser_for,
    read_batches,
    row,
    uuid_for,
    write_csv,
)

_USER_CAST_1 = field_column(Actor.USER, 1, ReplayField.CREATURES_CAST)
_OPPO_CAST_2 = field_column(Actor.OPPO, 2, ReplayField.CREATURES_CAST)


def test_code_table_codes_are_stable_and_shared_by_duplicates() -> None:
    first, second = binder_with_cards(), binder_with_cards()
    owlbear, morningstar = uuid_for(first, OWLBEAR), uuid_for(second, MORNINGSTAR)
    table = CardCodeTable([owlbear, owlbear])

    assert table.code_for(owlbear) == 0
    assert table.code_for(morningstar) == 1
    assert table.code_for(owlbear) == 0
    assert table.snapshot() == (owlbear, morningstar)


def test_field_columns_follow_the_exact_suffix_grammar() -> None:
    header = [
        "user_turn_3_user_creatures_killed_combat",
        "oppo_turn_12_creatures_cast",
        "user_turn_1_lands_played",
        "user_turn_x_creatures_cast",
        "deck_Owlbear",
    ]
    parser = ReplayDataChunkParser.from_header(
        ["draft_id", "match_number", "game_number", "num_turns", *header],
        binder_with_cards(),
        GameId.MTG,
    )

    assert parser._field_columns[ReplayField.USER_CREATURES_KILLED_COMBAT] == [
        FieldColumn("user_turn_3_user_creatures_killed_combat", Actor.USER, 3)
    ]
    assert parser._field_columns[ReplayField.CREATURES_CAST] == [
        FieldColumn("oppo_turn_12_creatures_cast", Actor.OPPO, 12)
    ]
    assert parser._field_columns[ReplayField.CARDS_TUTORED] == []


def test_needed_columns_skip_columns_no_metric_reads() -> None:
    needed = parser_for(binder_with_cards()).needed_columns()

    assert "user_turn_1_lands_played" not in needed
    assert "deck_Unmatched Card" not in needed
    assert _USER_CAST_1 in needed
    assert "num_turns" in needed


def test_a_missing_scalar_column_is_rejected() -> None:
    header = [column for column in HEADER if column != "num_turns"]

    with pytest.raises(ValueError, match="num_turns"):
        parser_for(binder_with_cards(), header)


def test_cells_split_into_one_entry_per_token(tmp_path: Path) -> None:
    binder = binder_with_cards()
    cells = {
        _USER_CAST_1: f"{OWLBEAR_ID}|{MORNINGSTAR_ID}|{OWLBEAR_ID}",
        _OPPO_CAST_2: MORNINGSTAR_ID,
    }

    chunk = parse_rows(tmp_path, binder, [row(), row(cells)])

    cast = chunk.events[ReplayField.CREATURES_CAST]
    uuids = [chunk.card_uuids[code] for code in cast.codes]
    owlbear, morningstar = uuid_for(binder, OWLBEAR), uuid_for(binder, MORNINGSTAR)
    assert uuids == [owlbear, morningstar, owlbear, morningstar]
    assert cast.rows.tolist() == [1, 1, 1, 1]
    assert cast.actors.tolist() == [0, 0, 0, 1]
    assert cast.turns.tolist() == [1, 1, 1, 2]


def test_empty_cells_have_no_entries(tmp_path: Path) -> None:
    chunk = parse_rows(tmp_path, binder_with_cards(), [row()])

    assert all(len(events) == 0 for events in chunk.events.values())


def test_float_formatted_tokens_match_like_integers(tmp_path: Path) -> None:
    binder = binder_with_cards()

    chunk = parse_rows(tmp_path, binder, [row({_USER_CAST_1: f"{OWLBEAR_ID}.0"})])

    (code,) = chunk.events[ReplayField.CREATURES_CAST].codes
    assert chunk.card_uuids[code] == uuid_for(binder, OWLBEAR)


def test_unknown_arena_ids_are_kept_as_unmatched(tmp_path: Path) -> None:
    chunk = parse_rows(tmp_path, binder_with_cards(), [row({_USER_CAST_1: UNKNOWN_ID})])

    assert chunk.events[ReplayField.CREATURES_CAST].codes.tolist() == [UNMATCHED]


def test_a_non_numeric_token_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        parse_rows(tmp_path, binder_with_cards(), [row({_USER_CAST_1: "abc"})])


def test_deck_codes_name_the_deck_columns(tmp_path: Path) -> None:
    binder = binder_with_cards()

    chunk = parse_rows(tmp_path, binder, [row(owlbear_deck=2)])

    named = tuple(chunk.card_uuids[code] for code in chunk.deck_codes)
    assert named == (uuid_for(binder, OWLBEAR), uuid_for(binder, MORNINGSTAR))
    assert chunk.deck.counts.tolist() == [[2, 0]]


def test_codes_stay_stable_across_chunks(tmp_path: Path) -> None:
    binder = binder_with_cards()
    rows = [row({_USER_CAST_1: OWLBEAR_ID}, draft_id=f"d{i}") for i in range(200)]
    csv_path = write_csv(tmp_path / "replays.csv", rows)
    parser = parser_for(binder)

    batches = read_batches(csv_path, parser, block_size=4096)
    chunks = [parser.parse(batch) for batch in batches]

    assert len(chunks) > 1
    first_codes = {
        chunk.events[ReplayField.CREATURES_CAST].codes[0] for chunk in chunks
    }
    assert len(first_codes) == 1
    assert chunks[0].card_uuids == chunks[-1].card_uuids[: len(chunks[0].card_uuids)]


def test_each_row_identifies_its_deck(tmp_path: Path) -> None:
    binder = binder_with_cards()

    chunk = parse_rows(tmp_path, binder, [row(owlbear_deck=1), row(morningstar_deck=1)])

    assert len(chunk.decks.decks) == 2
    assert chunk.decks.decks[0].name.startswith("replay_data draft1/1/1")


def test_the_afr_stx_layout_is_an_unsupported_layout() -> None:
    header = [
        "game_index" if column == "game_number" else column
        for column in HEADER
        if column != "match_number"
    ]

    with pytest.raises(UnsupportedCsvLayout, match="AFR/STX layout"):
        parser_for(binder_with_cards(), header)
