"""Tests for tutor_target_rate_metric.py's TutorTargetRateMetric, driven
through scan_game_csv as a real run drives it."""

from pathlib import Path

import pandas as pd

from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    VERSION,
    binder_with_cards,
    row,
    scan_into_frame,
    uuid_for,
)


def _metric(tmp_path: Path, name: str = "out.parquet") -> TutorTargetRateMetric:
    return TutorTargetRateMetric(VERSION, output_path=tmp_path / name)


def test_rate_is_times_tutored_over_times_in_deck(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    rows = [row(won=True, owlbear_deck=4, owlbear_tutored=1)] + [
        row(won=True, owlbear_deck=4) for _ in range(3)
    ]

    df = scan_into_frame(tmp_path, binder, rows, _metric(tmp_path))

    owlbear = str(uuid_for(binder, OWLBEAR))
    assert df.loc[owlbear, "tutor_target_rate"] == 0.25
    assert df.loc[owlbear, "sample_count"] == 4


def test_card_in_deck_but_never_tutored_gets_zero_rate_not_omitted(
    tmp_path: Path,
) -> None:
    binder = binder_with_cards([OWLBEAR])

    df = scan_into_frame(
        tmp_path, binder, [row(won=True, owlbear_deck=4)], _metric(tmp_path)
    )

    assert df.loc[str(uuid_for(binder, OWLBEAR)), "tutor_target_rate"] == 0.0


def test_card_never_in_any_deck_never_appears_in_output(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])

    df = scan_into_frame(
        tmp_path,
        binder,
        [row(won=True, owlbear_deck=4, owlbear_tutored=1)],
        _metric(tmp_path),
    )

    assert str(uuid_for(binder, MORNINGSTAR)) not in df.index


def test_tutored_but_not_in_deck_does_not_count(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    rows = [
        row(won=True, owlbear_deck=4, owlbear_tutored=1),
        row(won=True, owlbear_tutored=1),  # tutored, not in the deck
    ]

    df = scan_into_frame(tmp_path, binder, rows, _metric(tmp_path))

    owlbear = str(uuid_for(binder, OWLBEAR))
    assert df.loc[owlbear, "sample_count"] == 1
    assert df.loc[owlbear, "tutor_target_rate"] == 1.0


def test_many_small_chunks_tally_like_one(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
    rows = [
        row(
            won=True,
            owlbear_deck=1,
            owlbear_tutored=int(i % 4 == 0),
            morningstar_deck=1,
        )
        for i in range(60)
    ]

    one = scan_into_frame(tmp_path, binder, rows, _metric(tmp_path, "one.parquet"))
    many = scan_into_frame(
        tmp_path, binder, rows, _metric(tmp_path, "many.parquet"), block_size=256
    )

    pd.testing.assert_frame_equal(one.sort_index(), many.sort_index())
