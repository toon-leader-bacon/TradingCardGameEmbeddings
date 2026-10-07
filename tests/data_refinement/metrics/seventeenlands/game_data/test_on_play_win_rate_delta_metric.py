"""Tests for on_play_win_rate_delta_metric.py's OnPlayWinRateDeltaMetric,
driven through scan_game_csv as a real run drives it."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_refinement.seventeenlands.game_data.game_data_chunk import (
    GameZone,
)
from src.data_refinement.seventeenlands.zone_counts import ZoneCounts
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_delta_metric import (
    OnPlayWinRateDeltaMetric,
)
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    VERSION,
    binder_with_cards,
    chunk_with_zones,
    read_finished,
    row,
    scan_into_frame,
    uuid_for,
)


def _metric(tmp_path: Path, name: str = "out.parquet") -> OnPlayWinRateDeltaMetric:
    return OnPlayWinRateDeltaMetric(VERSION, output_path=tmp_path / name)


def test_computes_on_play_rate_minus_on_draw_rate_per_card(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    rows = [
        # On play: 2 wins / 2 games = 1.0. On draw: 0 wins / 1 game = 0.0.
        row(on_play=True, won=True, owlbear_deck=4),
        row(on_play=True, won=True, owlbear_deck=4),
        row(on_play=False, won=False, owlbear_deck=4),
    ]

    df = scan_into_frame(tmp_path, binder, rows, _metric(tmp_path))

    assert df.loc[str(uuid_for(binder, OWLBEAR)), "on_play_win_rate_delta"] == 1.0


def test_delta_is_none_when_card_never_seen_on_one_side(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])

    df = scan_into_frame(
        tmp_path,
        binder,
        [row(on_play=True, won=True, owlbear_deck=4)],
        _metric(tmp_path),
    )

    # A written None reads back as NaN in a float64 column
    assert pd.isna(df.loc[str(uuid_for(binder, OWLBEAR)), "on_play_win_rate_delta"])


def test_sample_count_sums_both_sides(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    rows = [
        row(on_play=True, won=True, owlbear_deck=4),
        row(on_play=True, won=False, owlbear_deck=4),
        row(on_play=False, won=True, owlbear_deck=4),
    ]

    df = scan_into_frame(tmp_path, binder, rows, _metric(tmp_path))

    assert df.loc[str(uuid_for(binder, OWLBEAR)), "sample_count"] == 3


def test_only_deck_present_cards_are_tallied(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])

    df = scan_into_frame(
        tmp_path,
        binder,
        [row(on_play=True, won=True, owlbear_deck=4)],
        _metric(tmp_path),
    )

    assert str(uuid_for(binder, MORNINGSTAR)) not in df.index


def test_many_small_chunks_tally_like_one(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
    rows = [
        row(won=i % 3 == 0, on_play=i % 2 == 0, owlbear_deck=i % 2, morningstar_deck=1)
        for i in range(60)
    ]

    one = scan_into_frame(tmp_path, binder, rows, _metric(tmp_path, "one.parquet"))
    many = scan_into_frame(
        tmp_path, binder, rows, _metric(tmp_path, "many.parquet"), block_size=256
    )

    pd.testing.assert_frame_equal(one.sort_index(), many.sort_index())


def test_two_columns_naming_one_card_both_count(tmp_path: Path) -> None:
    card = uuid_for(binder_with_cards([OWLBEAR]), OWLBEAR)
    deck = ZoneCounts((card, card), np.array([[1, 1]], np.int16))
    metric = _metric(tmp_path)

    metric.accumulate(chunk_with_zones({GameZone.DECK: deck}, won=[True]))

    df = read_finished(metric).set_index("nocab_uuid")
    assert df.loc[str(card), "sample_count"] == 2


def test_no_rows_writes_an_empty_file(tmp_path: Path) -> None:
    df = scan_into_frame(
        tmp_path, binder_with_cards([OWLBEAR]), [], _metric(tmp_path), index=None
    )

    assert len(df) == 0
