"""Tests for game_data_chunk.py: ZoneCounts, GameDataChunk's invariants,
and source_frame_of."""

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
    ZoneCounts,
    source_frame_of,
)


def _empty_zone(rows: int) -> ZoneCounts:
    return ZoneCounts(card_uuids=(), counts=np.zeros((rows, 0), np.int16))


def _chunk(
    rows: int = 2,
    zones: dict | None = None,
    num_turns_rows: int | None = None,
    source_frame: pd.DataFrame | None = None,
) -> GameDataChunk:
    return GameDataChunk(
        zones=(
            zones
            if zones is not None
            else {zone: _empty_zone(rows) for zone in GameZone}
        ),
        won=np.zeros(rows, np.bool_),
        on_play=np.zeros(rows, np.bool_),
        num_turns=np.zeros(
            num_turns_rows if num_turns_rows is not None else rows, np.int32
        ),
        source_frame=source_frame,
    )


def test_present_marks_cells_with_at_least_one_copy() -> None:
    zone = ZoneCounts(
        card_uuids=(uuid4(), uuid4()), counts=np.array([[0, 2], [1, 0]], np.int16)
    )

    assert zone.present().tolist() == [[False, True], [True, False]]


def test_len_is_the_row_count() -> None:
    assert len(_chunk(rows=3)) == 3


def test_a_missing_zone_is_rejected() -> None:
    zones = {zone: _empty_zone(2) for zone in GameZone if zone is not GameZone.TUTORED}

    with pytest.raises(ValueError, match="TUTORED"):
        _chunk(zones=zones)


def test_fields_with_different_row_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="num_turns"):
        _chunk(rows=2, num_turns_rows=3)


def test_a_source_frame_with_a_different_row_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="source_frame"):
        _chunk(rows=2, source_frame=pd.DataFrame({"won": [True]}))


def test_source_frame_of_returns_the_frame() -> None:
    frame = pd.DataFrame({"won": [True, False]})

    assert source_frame_of(_chunk(rows=2, source_frame=frame)) is frame


def test_source_frame_of_refuses_a_chunk_without_one() -> None:
    with pytest.raises(ValueError, match="keep_source_frame=False"):
        source_frame_of(_chunk(rows=2))
