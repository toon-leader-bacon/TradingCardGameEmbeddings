"""Tests for draft_data_chunk_parser.py's DraftDataChunkParser and the
DraftDataChunk it builds."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pytest

from src.schema.game_id import GameId

from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk import (
    NO_PICK,
    DraftDataChunk,
    DraftStratum,
)
from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk_parser import (
    DraftDataChunkParser,
)
from tests.data_refinement.metrics.seventeenlands.draft_data._draft_fixtures import (
    BOLT,
    HEADER,
    MORNINGSTAR,
    OWLBEAR,
    PICK_TWO_HEADER,
    binder_with_cards,
    parser_for,
    row,
    uuid_for,
    write_csv,
)

_BINDER = binder_with_cards([OWLBEAR, MORNINGSTAR, BOLT])


def _parse(
    tmp_path: Path, rows: list[dict], header: list[str] = HEADER
) -> DraftDataChunk:
    parser = parser_for(_BINDER, header)
    reader = pa_csv.open_csv(
        write_csv(tmp_path / "picks.csv", rows, header),
        convert_options=pa_csv.ConvertOptions(
            include_columns=parser.needed_columns(),
            column_types=parser.column_types(),
        ),
    )
    return parser.parse(reader.read_next_batch())


def test_a_header_without_a_scalar_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="pick_number"):
        DraftDataChunkParser.from_header(
            [c for c in HEADER if c != "pick_number"], _BINDER, GameId.MTG
        )


def test_column_types_read_card_counts_as_float32() -> None:
    types = parser_for(_BINDER).column_types()

    assert types[f"pack_card_{OWLBEAR}"] == pa.float32()
    assert types[f"pool_{BOLT}"] == pa.float32()
    assert types["pick_number"] == pa.int64()
    assert "pick_2" not in types


def test_needed_columns_include_pick_2_only_when_present() -> None:
    assert "pick_2" not in parser_for(_BINDER).needed_columns()
    assert "pick_2" in parser_for(_BINDER, PICK_TWO_HEADER).needed_columns()


def test_parse_builds_pack_and_pool_matrices(tmp_path: Path) -> None:
    chunk = _parse(tmp_path, [row(OWLBEAR, owlbear=1, morningstar=2, bolt_pool=1)])

    assert chunk.pack.counts.tolist() == [[1, 2]]
    assert chunk.pool.counts.tolist() == [[0, 1]]
    assert chunk.pack.card_uuids == (
        uuid_for(_BINDER, OWLBEAR),
        uuid_for(_BINDER, MORNINGSTAR),
    )


def test_float_formatted_counts_read_as_int16(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "picks.csv", [row(OWLBEAR, owlbear=1)])
    path.write_text(path.read_text().replace(",1,", ",1.0,"))
    parser = parser_for(_BINDER)
    reader = pa_csv.open_csv(
        path,
        convert_options=pa_csv.ConvertOptions(
            include_columns=parser.needed_columns(),
            column_types=parser.column_types(),
        ),
    )

    chunk = parser.parse(reader.read_next_batch())

    assert chunk.pack.counts.dtype == np.int16
    assert chunk.pack.counts.tolist()[0][0] == 1


def test_picked_marks_the_taken_pack_cell(tmp_path: Path) -> None:
    chunk = _parse(tmp_path, [row(MORNINGSTAR, owlbear=1, morningstar=1)])

    assert chunk.picked().tolist() == [[False, True]]
    assert chunk.picks.first_uuids.tolist() == [str(uuid_for(_BINDER, MORNINGSTAR))]


def test_an_unmatched_pick_codes_as_no_pick_with_no_uuid(tmp_path: Path) -> None:
    chunk = _parse(tmp_path, [row("Nonexistent Card", owlbear=1)])

    assert chunk.picks.first_codes.tolist() == [NO_PICK]
    assert chunk.picks.first_uuids.tolist() == [None]
    assert not chunk.picked().any()


def test_a_pick_two_row_marks_both_picks_and_the_flag(tmp_path: Path) -> None:
    rows = [
        row(OWLBEAR, owlbear=1, morningstar=1, pick_2=MORNINGSTAR),
        row(OWLBEAR, owlbear=1),
        row(OWLBEAR, owlbear=1, pick_2="Unmatched Card"),
    ]

    chunk = _parse(tmp_path, rows, PICK_TWO_HEADER)

    assert chunk.picks.is_pick_two.tolist() == [True, False, True]
    assert chunk.picked().tolist() == [[True, True], [True, False], [True, False]]


def test_without_pick_2_no_row_is_a_pick_two(tmp_path: Path) -> None:
    chunk = _parse(tmp_path, [row(OWLBEAR, owlbear=1)])

    assert not chunk.picks.is_pick_two.any()
    assert chunk.picks.second_codes.tolist() == [NO_PICK]


def test_strata_and_an_empty_rank(tmp_path: Path) -> None:
    chunk = _parse(tmp_path, [row(OWLBEAR, pack_number=2, pick_number=5, rank="")])

    assert chunk.stratum(DraftStratum.PACK_NUMBER).tolist() == [2]
    assert chunk.stratum(DraftStratum.PICK_NUMBER).tolist() == [5]
    assert chunk.stratum(DraftStratum.RANK).tolist() == [""]


def test_a_null_scalar_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "picks.csv"
    write_csv(path, [row(OWLBEAR, owlbear=1)])
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace("draft1", "")
    path.write_text("\n".join(lines) + "\n")
    parser = parser_for(_BINDER)
    reader = pa_csv.open_csv(
        path,
        convert_options=pa_csv.ConvertOptions(
            include_columns=parser.needed_columns(),
            column_types=parser.column_types(),
            strings_can_be_null=True,
        ),
    )

    with pytest.raises(ValueError, match="'draft_id' is null"):
        parser.parse(reader.read_next_batch())


def test_a_chunk_with_mismatched_row_counts_is_rejected(tmp_path: Path) -> None:
    chunk = _parse(tmp_path, [row(OWLBEAR, owlbear=1)])

    with pytest.raises(ValueError, match="draft_id has 2 rows"):
        DraftDataChunk(
            pack=chunk.pack,
            pool=chunk.pool,
            pack_column_codes=chunk.pack_column_codes,
            picks=chunk.picks,
            draft_id=np.array(["a", "b"], np.object_),
            pack_number=chunk.pack_number,
            pick_number=chunk.pick_number,
            rank=chunk.rank,
        )
