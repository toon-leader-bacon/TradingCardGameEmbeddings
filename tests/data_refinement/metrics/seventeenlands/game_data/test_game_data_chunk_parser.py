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


def test_needed_columns_skips_unmatched_columns() -> None:
    parser = parser_for(binder_with_cards([OWLBEAR, MORNINGSTAR]))

    needed = parser.needed_columns()

    assert "deck_Unmatched Card" not in needed
    assert {
        "draft_id",
        "match_number",
        "game_number",
        "won",
        "on_play",
        "num_turns",
        "rank",
        f"deck_{OWLBEAR}",
    } <= set(needed)


def test_column_types_reads_card_counts_as_float32_and_scalars_typed() -> None:
    types = parser_for(binder_with_cards([OWLBEAR])).column_types()

    assert types[f"deck_{OWLBEAR}"] == pa.float32()
    assert types["won"] == pa.bool_()
    assert types["num_turns"] == pa.int32()
    assert types["rank"] == pa.string()
    assert types["match_number"] == pa.int64()


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


def test_parse_reads_the_game_keys_and_rank(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    csv_path = write_csv(
        tmp_path / "g.csv",
        [
            row(won=True, draft_id="d1", match_number=2, game_number=3),
            row(won=True, rank="mythic"),
        ],
    )

    chunk = parse_one(csv_path, parser_for(binder))

    assert chunk.keys.draft_id.tolist() == ["d1", "draft1"]
    assert chunk.keys.match_number.tolist() == [2, 1]
    assert chunk.keys.game_number.tolist() == [3, 1]
    assert chunk.rank.tolist() == ["gold", "mythic"]


def test_an_empty_rank_reads_as_an_empty_string(tmp_path: Path) -> None:
    # Trad and Sealed events leave every rank cell empty
    csv_path = write_csv(tmp_path / "g.csv", [row(won=True, rank="")] * 2)

    chunk = parse_one(csv_path, parser_for(binder_with_cards([OWLBEAR])))

    assert chunk.rank.tolist() == ["", ""]


def test_parse_identifies_each_rows_deck(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
    csv_path = write_csv(
        tmp_path / "g.csv",
        [
            row(won=True, owlbear_deck=2),
            row(won=True, morningstar_deck=1),
            row(won=True, owlbear_deck=4, game_number=2),
        ],
    )

    chunk = parse_one(csv_path, parser_for(binder))

    assert chunk.decks.row_deck.tolist() == [0, 1, 0]
    assert chunk.decks.decks[0].card_nocab_uuids == [uuid_for(binder, OWLBEAR)]


def test_an_older_export_without_match_number_reads_it_as_zero(
    tmp_path: Path,
) -> None:
    # AFR/KHM/MID/STX/VOW exports have no match_number column
    binder = binder_with_cards([OWLBEAR])
    header = [column for column in HEADER if column != "match_number"]
    rows = [row(won=True, game_number=1), row(won=False, game_number=2)]
    for each in rows:
        del each["match_number"]
    parser = parser_for(binder, header=header)

    chunk = parse_one(write_csv(tmp_path / "g.csv", rows, header=header), parser)

    assert "match_number" not in parser.needed_columns()
    assert "match_number" not in parser.column_types()
    assert chunk.keys.match_number.tolist() == [0, 0]
    assert chunk.keys.game_number.tolist() == [1, 2]


def test_float_formatted_counts_read_as_int16(tmp_path: Path) -> None:
    # Some exports (e.g. BRO.PremierDraft) write counts as "1.0"
    binder = binder_with_cards([OWLBEAR])
    csv_path = write_csv(tmp_path / "g.csv", [row(won=True, owlbear_deck=2)])
    csv_path.write_text(
        csv_path.read_text().replace(",2,", ",2.0,").replace(",0,", ",0.0,")
    )

    chunk = parse_one(csv_path, parser_for(binder))

    deck = chunk.zones[GameZone.DECK]
    assert deck.counts.dtype.name == "int16"
    assert deck.counts.tolist() == [[2]]
