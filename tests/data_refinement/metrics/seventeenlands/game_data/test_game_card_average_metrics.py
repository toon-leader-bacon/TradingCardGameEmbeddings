"""Tests for game_card_average_metric.py's GameCardAverageMetric, covered
through its game_card_average_metrics.py concretes and driven through
scan_game_csv, as a real run drives them."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metrics import (
    DrawnWinRateMetric,
    OpeningHandWinRateMetric,
    WinRateWhenInDeckMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameZone,
    ZoneCounts,
)
from src.data_refinement.metrics.version_metadata import read_version_metadata
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    VERSION,
    binder_with_cards,
    chunk_with_zones,
    row,
    scan_into_frame,
    uuid_for,
)


class TestWinRateWhenInDeckMetric:
    def test_averages_won_across_games_with_card_in_deck(self, tmp_path: Path) -> None:
        binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
        metric = WinRateWhenInDeckMetric(VERSION, output_path=tmp_path / "out.parquet")

        df = scan_into_frame(
            tmp_path,
            binder,
            [row(won=True, owlbear_deck=4), row(won=False, owlbear_deck=4)],
            metric,
        )

        owlbear = str(uuid_for(binder, OWLBEAR))
        assert df.loc[owlbear, "win_rate_when_in_deck"] == 0.5
        assert df.loc[owlbear, "sample_count"] == 2

    def test_a_card_never_in_a_deck_has_no_row(self, tmp_path: Path) -> None:
        binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
        metric = WinRateWhenInDeckMetric(VERSION, output_path=tmp_path / "out.parquet")

        df = scan_into_frame(tmp_path, binder, [row(won=True, owlbear_deck=4)], metric)

        assert str(uuid_for(binder, MORNINGSTAR)) not in df.index

    def test_sample_count_counts_qualifying_games(self, tmp_path: Path) -> None:
        binder = binder_with_cards([OWLBEAR])
        metric = WinRateWhenInDeckMetric(VERSION, output_path=tmp_path / "out.parquet")

        rows = [
            row(won=True, owlbear_deck=4),
            row(won=True, owlbear_deck=4),
            row(won=True),
        ]
        df = scan_into_frame(tmp_path, binder, rows, metric)

        assert df.loc[str(uuid_for(binder, OWLBEAR)), "sample_count"] == 2

    def test_many_small_chunks_tally_like_one(self, tmp_path: Path) -> None:
        binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
        rows = [
            row(won=i % 3 == 0, owlbear_deck=i % 2, morningstar_deck=1)
            for i in range(60)
        ]
        one = scan_into_frame(
            tmp_path,
            binder,
            rows,
            WinRateWhenInDeckMetric(VERSION, tmp_path / "one.parquet"),
        )
        many = scan_into_frame(
            tmp_path,
            binder,
            rows,
            WinRateWhenInDeckMetric(VERSION, tmp_path / "many.parquet"),
            block_size=256,
        )

        pd.testing.assert_frame_equal(one.sort_index(), many.sort_index())

    def test_output_carries_the_run_version(self, tmp_path: Path) -> None:
        binder = binder_with_cards([OWLBEAR])
        metric = WinRateWhenInDeckMetric(VERSION, output_path=tmp_path / "out.parquet")

        scan_into_frame(tmp_path, binder, [row(won=True, owlbear_deck=1)], metric)

        metadata = read_version_metadata(tmp_path / "out.parquet")
        assert metadata is not None
        assert metadata.card_binder_version == "test-version"


class TestOpeningHandWinRateMetric:
    def test_averages_won_across_games_with_card_in_opening_hand(
        self, tmp_path: Path
    ) -> None:
        binder = binder_with_cards([OWLBEAR])
        metric = OpeningHandWinRateMetric(VERSION, output_path=tmp_path / "out.parquet")

        rows = [
            row(won=True, owlbear_deck=4, owlbear_opening_hand=1),
            row(won=False, owlbear_deck=4, owlbear_opening_hand=1),
            row(won=True, owlbear_deck=4),  # in deck, not in the opening hand
        ]
        df = scan_into_frame(tmp_path, binder, rows, metric)

        owlbear = str(uuid_for(binder, OWLBEAR))
        assert df.loc[owlbear, "opening_hand_win_rate"] == 0.5
        assert df.loc[owlbear, "sample_count"] == 2


class TestDrawnWinRateMetric:
    def test_counts_a_card_drawn_after_the_opening_hand(self, tmp_path: Path) -> None:
        binder = binder_with_cards([OWLBEAR])
        metric = DrawnWinRateMetric(VERSION, output_path=tmp_path / "out.parquet")

        df = scan_into_frame(
            tmp_path, binder, [row(won=True, owlbear_deck=4, owlbear_drawn=1)], metric
        )

        owlbear = str(uuid_for(binder, OWLBEAR))
        assert df.loc[owlbear, "drawn_win_rate"] == 1.0
        assert df.loc[owlbear, "sample_count"] == 1


def test_an_empty_csv_writes_a_zero_row_file_with_the_full_schema(
    tmp_path: Path,
) -> None:
    binder = binder_with_cards([OWLBEAR])
    metric = WinRateWhenInDeckMetric(VERSION, output_path=tmp_path / "out.parquet")

    df = scan_into_frame(tmp_path, binder, [], metric)

    assert len(df) == 0
    assert list(df.reset_index().columns) == [
        "nocab_uuid",
        "win_rate_when_in_deck",
        "sample_count",
    ]


def test_two_columns_naming_one_card_both_count(tmp_path: Path) -> None:
    owlbear_uuid = uuid_for(binder_with_cards([OWLBEAR]), OWLBEAR)
    metric = WinRateWhenInDeckMetric(VERSION, output_path=tmp_path / "out.parquet")
    deck = ZoneCounts(
        card_uuids=(owlbear_uuid, owlbear_uuid),
        counts=np.array([[1, 1], [1, 0]], np.int16),
    )

    metric.accumulate(chunk_with_zones({GameZone.DECK: deck}, won=[True, False]))
    df = pd.read_parquet(metric.finalize()).set_index("nocab_uuid")

    # The row implementation counted each matching column once per game
    assert df.loc[str(owlbear_uuid), "sample_count"] == 3
    assert df.loc[str(owlbear_uuid), "win_rate_when_in_deck"] == pytest.approx(2 / 3)


def test_a_chunk_with_a_different_column_layout_is_rejected(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
    metric = WinRateWhenInDeckMetric(VERSION, output_path=tmp_path / "out.parquet")
    first = ZoneCounts((uuid_for(binder, OWLBEAR),), np.array([[1]], np.int16))
    second = ZoneCounts((uuid_for(binder, MORNINGSTAR),), np.array([[1]], np.int16))

    metric.accumulate(chunk_with_zones({GameZone.DECK: first}, won=[True]))
    with pytest.raises(ValueError, match="differ from the first chunk"):
        metric.accumulate(chunk_with_zones({GameZone.DECK: second}, won=[True]))
