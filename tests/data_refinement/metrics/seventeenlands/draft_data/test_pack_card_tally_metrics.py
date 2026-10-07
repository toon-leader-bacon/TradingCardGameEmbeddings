"""Tests for pack_card_tally_metric.py's PackCardTallyMetric, through its
pack_card_tally_metrics.py concretes, driven through scan_draft_csv."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metrics import (
    CardTakeRateMetric,
    FirstPickRateMetric,
    RankStratifiedTakeRateMetric,
)
from tests.data_refinement.metrics.seventeenlands.draft_data._draft_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    PICK_TWO_HEADER,
    VERSION,
    binder_with_cards,
    row,
    scan_into_frame,
    uuid_for,
)

_BINDER = binder_with_cards([OWLBEAR, MORNINGSTAR])


def _by_key(frame: pd.DataFrame, keys: list[str]) -> dict[tuple, dict]:
    return {
        tuple(record[k] for k in keys): record for record in frame.to_dict("records")
    }


class TestCardTakeRateMetric:
    def test_take_rate_tallies_per_pack_and_pick_number(self, tmp_path: Path) -> None:
        rows = [
            row(OWLBEAR, owlbear=1, morningstar=1),
            row(MORNINGSTAR, owlbear=1, morningstar=1),
        ]
        metric = CardTakeRateMetric(VERSION, tmp_path / "out.parquet")

        frame = scan_into_frame(tmp_path, _BINDER, rows, metric)

        by_key = _by_key(frame, ["nocab_uuid", "pack_number", "pick_number"])
        owlbear = by_key[(str(uuid_for(_BINDER, OWLBEAR)), 0, 0)]
        assert owlbear["take_rate"] == 0.5
        assert owlbear["sample_count"] == 2

    def test_absent_card_is_never_tallied(self, tmp_path: Path) -> None:
        metric = CardTakeRateMetric(VERSION, tmp_path / "out.parquet")

        frame = scan_into_frame(tmp_path, _BINDER, [row(OWLBEAR, owlbear=1)], metric)

        assert set(frame["nocab_uuid"]) == {str(uuid_for(_BINDER, OWLBEAR))}

    def test_different_pack_and_pick_numbers_are_separate_keys(
        self, tmp_path: Path
    ) -> None:
        rows = [
            row(OWLBEAR, owlbear=1, pick_number=0),
            row(MORNINGSTAR, owlbear=1, morningstar=1, pick_number=1),
        ]
        metric = CardTakeRateMetric(VERSION, tmp_path / "out.parquet")

        frame = scan_into_frame(tmp_path, _BINDER, rows, metric)

        owlbear = frame[frame["nocab_uuid"] == str(uuid_for(_BINDER, OWLBEAR))]
        assert sorted(owlbear["pick_number"]) == [0, 1]
        assert sorted(owlbear["take_rate"]) == [0.0, 1.0]

    def test_an_unmatched_pick_name_takes_nothing(self, tmp_path: Path) -> None:
        metric = CardTakeRateMetric(VERSION, tmp_path / "out.parquet")

        frame = scan_into_frame(
            tmp_path, _BINDER, [row("Nonexistent Card", owlbear=1)], metric
        )

        assert frame["take_rate"].tolist() == [0.0]

    def test_a_pick_two_rows_second_pick_counts_as_taken(self, tmp_path: Path) -> None:
        rows = [row(OWLBEAR, owlbear=1, morningstar=1, pick_2=MORNINGSTAR)]
        metric = CardTakeRateMetric(VERSION, tmp_path / "out.parquet")

        frame = scan_into_frame(tmp_path, _BINDER, rows, metric, PICK_TWO_HEADER)

        assert frame["take_rate"].tolist() == [1.0, 1.0]

    def test_many_small_chunks_tally_like_one(self, tmp_path: Path) -> None:
        rows = [
            row(
                OWLBEAR if i % 3 else MORNINGSTAR,
                owlbear=1,
                morningstar=i % 2,
                pick_number=i % 4,
            )
            for i in range(80)
        ]
        one = scan_into_frame(
            tmp_path,
            _BINDER,
            rows,
            CardTakeRateMetric(VERSION, tmp_path / "one.parquet"),
        )
        many = scan_into_frame(
            tmp_path,
            _BINDER,
            rows,
            CardTakeRateMetric(VERSION, tmp_path / "many.parquet"),
            block_size=256,
        )

        keys = ["nocab_uuid", "pack_number", "pick_number"]
        pd.testing.assert_frame_equal(
            one.sort_values(keys).reset_index(drop=True),
            many.sort_values(keys).reset_index(drop=True),
        )


class TestFirstPickRateMetric:
    def test_only_pack_zero_pick_zero_rows_are_tallied(self, tmp_path: Path) -> None:
        rows = [
            row(OWLBEAR, owlbear=1),
            row(MORNINGSTAR, owlbear=1, morningstar=1, pick_number=1),
        ]
        metric = FirstPickRateMetric(VERSION, tmp_path / "out.parquet")

        frame = scan_into_frame(tmp_path, _BINDER, rows, metric)

        assert frame.to_dict("records") == [
            {
                "nocab_uuid": str(uuid_for(_BINDER, OWLBEAR)),
                "take_rate": 1.0,
                "sample_count": 1,
            }
        ]


class TestRankStratifiedTakeRateMetric:
    def test_separate_ranks_are_tallied_independently(self, tmp_path: Path) -> None:
        rows = [
            row(OWLBEAR, owlbear=1, rank="gold"),
            row(MORNINGSTAR, owlbear=1, morningstar=1, rank="mythic"),
        ]
        metric = RankStratifiedTakeRateMetric(VERSION, tmp_path / "out.parquet")

        frame = scan_into_frame(tmp_path, _BINDER, rows, metric)

        owlbear = frame[frame["nocab_uuid"] == str(uuid_for(_BINDER, OWLBEAR))]
        assert dict(zip(owlbear["rank"], owlbear["take_rate"])) == {
            "gold": 1.0,
            "mythic": 0.0,
        }

    def test_rankless_rows_are_not_tallied(self, tmp_path: Path) -> None:
        rows = [row(OWLBEAR, owlbear=1, rank=""), row(OWLBEAR, owlbear=1)]
        metric = RankStratifiedTakeRateMetric(VERSION, tmp_path / "out.parquet")

        frame = scan_into_frame(tmp_path, _BINDER, rows, metric)

        assert frame["rank"].tolist() == ["gold"]
        assert frame["sample_count"].tolist() == [1]


def test_no_rows_writes_the_count_schema(tmp_path: Path) -> None:
    metric = CardTakeRateMetric(VERSION, tmp_path / "out.parquet")

    scan_into_frame(tmp_path, _BINDER, [], metric)

    table = pq.read_table(tmp_path / "out.parquet")
    assert table.num_rows == 0
    assert table.column_names == [
        "nocab_uuid",
        "pack_number",
        "pick_number",
        "in_pack",
        "picked",
    ]


def test_output_stems_are_distinct() -> None:
    stems = {
        CardTakeRateMetric.OUTPUT_STEM,
        FirstPickRateMetric.OUTPUT_STEM,
        RankStratifiedTakeRateMetric.OUTPUT_STEM,
    }
    assert len(stems) == 3
