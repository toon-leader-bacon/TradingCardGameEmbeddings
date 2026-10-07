"""Tests for tutor_choice_rate_metric.py's TutorChoiceRateMetric, driven
through scan_game_csv as a real run drives it."""

from pathlib import Path

from src.data_refinement.metrics.seventeenlands.game_data.tutor_choice_rate_metric import (
    TutorChoiceRateMetric,
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

_BINDER = binder_with_cards([OWLBEAR, MORNINGSTAR])


def _scan(tmp_path: Path, rows: list[dict]):
    metric = TutorChoiceRateMetric(VERSION, output_path=tmp_path / "out.parquet")
    return scan_into_frame(tmp_path, _BINDER, rows, metric)


def test_only_games_with_a_tutor_count(tmp_path: Path) -> None:
    rows = [
        row(won=True, owlbear_deck=1, owlbear_tutored=1),
        row(won=True, owlbear_deck=1),  # no tutor this game: not counted
    ]

    df = _scan(tmp_path, rows)

    owlbear = str(uuid_for(_BINDER, OWLBEAR))
    assert df.loc[owlbear, "tutor_choice_rate"] == 1.0
    assert df.loc[owlbear, "sample_count"] == 1


def test_a_deck_card_not_picked_by_the_tutor_counts_toward_the_denominator(
    tmp_path: Path,
) -> None:
    rows = [
        row(won=True, owlbear_deck=1, morningstar_deck=1, owlbear_tutored=1),
        row(won=True, morningstar_deck=1),  # no tutor: not counted
    ]

    df = _scan(tmp_path, rows)

    morningstar = str(uuid_for(_BINDER, MORNINGSTAR))
    assert df.loc[morningstar, "tutor_choice_rate"] == 0.0
    assert df.loc[morningstar, "sample_count"] == 1


def test_a_card_only_in_tutorless_games_has_no_row(tmp_path: Path) -> None:
    df = _scan(tmp_path, [row(won=True, morningstar_deck=1)])

    assert str(uuid_for(_BINDER, MORNINGSTAR)) not in df.index


def test_the_sideboard_is_not_counted(tmp_path: Path) -> None:
    rows = [row(won=True, owlbear_sideboard=1, owlbear_tutored=1)]

    df = _scan(tmp_path, rows)

    assert str(uuid_for(_BINDER, OWLBEAR)) not in df.index
