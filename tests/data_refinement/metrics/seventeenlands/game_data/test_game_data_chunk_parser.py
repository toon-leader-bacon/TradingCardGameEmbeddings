"""Tests for game_data_chunk_parser.py's GameDataChunkParser."""

from pathlib import Path

import pyarrow as pa
import pytest

from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameZone,
)
from src.schema.game_id import GameId
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    HEADER,
    MORNINGSTAR,
    OWLBEAR,
    binder_with_cards,
    parse_one,
    parser_for,
    row,
    uuid_for,
    write_csv,
)


def test_a_header_without_the_scalar_columns_is_rejected() -> None:
    binder = binder_with_cards([OWLBEAR])

    with pytest.raises(ValueError, match="num_turns"):
        parser_for(binder, header=[c for c in HEADER if c != "num_turns"])


def test_needed_columns_skips_unmatched_and_unused_columns() -> None:
    parser = parser_for(binder_with_cards([OWLBEAR, MORNINGSTAR]))

    needed = parser.needed_columns()

    assert "rank" not in needed
    assert "deck_Unmatched Card" not in needed
    assert {"won", "on_play", "num_turns", f"deck_{OWLBEAR}"} <= set(needed)


def test_needed_columns_is_every_column_when_keeping_a_source_frame() -> None:
    parser = parser_for(binder_with_cards([OWLBEAR]), keep_source_frame=True)

    assert parser.needed_columns() == HEADER


def test_column_types_reads_card_counts_as_int16_and_scalars_typed() -> None:
    types = parser_for(binder_with_cards([OWLBEAR])).column_types()

    assert types[f"deck_{OWLBEAR}"] == pa.int16()
    assert types["won"] == pa.bool_()
    assert types["num_turns"] == pa.int32()


def test_parse_builds_one_count_matrix_per_zone(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
    csv_path = write_csv(
        tmp_path / "g.csv",
        [
            row(won=True, owlbear_deck=2, morningstar_deck=1),
            row(won=False, num_turns=11),
        ],
    )

    chunk = parse_one(csv_path, parser_for(binder))

    deck = chunk.zones[GameZone.DECK]
    assert deck.card_uuids == (uuid_for(binder, OWLBEAR), uuid_for(binder, MORNINGSTAR))
    assert deck.counts.tolist() == [[2, 1], [0, 0]]
    assert chunk.won.tolist() == [True, False]
    assert chunk.num_turns.tolist() == [8, 11]
    assert chunk.zones[GameZone.SIDEBOARD].card_uuids == (uuid_for(binder, OWLBEAR),)
    assert chunk.source_frame is None


def test_a_zone_with_no_matched_columns_is_empty(tmp_path: Path) -> None:
    binder = binder_with_cards([MORNINGSTAR])
    csv_path = write_csv(tmp_path / "g.csv", [row(won=True)])

    chunk = parse_one(csv_path, parser_for(binder))

    # Only Owlbear has tutored_/sideboard_ columns, and it isn't in the binder
    assert chunk.zones[GameZone.TUTORED].counts.shape == (1, 0)


def test_a_null_count_cell_is_zero(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    rows = [row(won=True, owlbear_deck=3)]
    rows[0][f"deck_{OWLBEAR}"] = None
    csv_path = write_csv(tmp_path / "g.csv", rows)

    chunk = parse_one(csv_path, parser_for(binder))

    assert chunk.zones[GameZone.DECK].counts.tolist() == [[0]]


def test_a_null_scalar_is_rejected_with_its_column_and_row(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    rows = [row(won=True), row(won=False)]
    rows[1]["won"] = None
    csv_path = write_csv(tmp_path / "g.csv", rows)

    with pytest.raises(ValueError, match="'won' is null at batch row 1"):
        parse_one(csv_path, parser_for(binder))


def test_zone_columns_match_the_row_implementations_matching(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
    csv_path = write_csv(tmp_path / "g.csv", [row(won=True)])
    game_columns = GameCardColumns.from_header(HEADER, binder, GameId.MTG)

    chunk = parse_one(csv_path, parser_for(binder))

    assert chunk.zones[GameZone.DECK].card_uuids == tuple(
        uuid for _, uuid in game_columns.deck_columns
    )
    assert chunk.zones[GameZone.OPENING_HAND].card_uuids == tuple(
        uuid for _, uuid in game_columns.opening_hand_columns
    )


def test_source_frame_is_kept_when_asked(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    csv_path = write_csv(tmp_path / "g.csv", [row(won=True, owlbear_deck=2)])

    chunk = parse_one(csv_path, parser_for(binder, keep_source_frame=True))

    assert chunk.source_frame is not None
    assert chunk.source_frame.loc[0, "rank"] == "gold"
    assert chunk.source_frame.loc[0, f"deck_{OWLBEAR}"] == 2
