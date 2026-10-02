"""Tests for game_length_association_metric.py's
GameLengthAssociationMetric, driven through scan_game_csv."""

from pathlib import Path

from src.data_refinement.metrics.seventeenlands.game_data.game_length_association_metric import (
    GameLengthAssociationMetric,
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


def test_subtracts_the_format_wide_average_from_each_cards_average(
    tmp_path: Path,
) -> None:
    binder = binder_with_cards([OWLBEAR, MORNINGSTAR])
    metric = GameLengthAssociationMetric(VERSION, output_path=tmp_path / "out.parquet")

    rows = [
        row(won=True, num_turns=12, owlbear_deck=1),
        row(won=True, num_turns=4, morningstar_deck=1),
    ]
    df = scan_into_frame(tmp_path, binder, rows, metric)

    assert df.loc[str(uuid_for(binder, OWLBEAR)), "game_length_association"] == 4.0
    assert df.loc[str(uuid_for(binder, MORNINGSTAR)), "game_length_association"] == -4.0


def test_the_baseline_counts_games_with_no_tallied_card(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    metric = GameLengthAssociationMetric(VERSION, output_path=tmp_path / "out.parquet")

    rows = [row(won=True, num_turns=5, owlbear_deck=1), row(won=True, num_turns=15)]
    df = scan_into_frame(tmp_path, binder, rows, metric)

    # Owlbear's 5 turns minus the baseline (5 + 15) / 2 = 10
    assert df.loc[str(uuid_for(binder, OWLBEAR)), "game_length_association"] == -5.0


def test_only_the_designated_steps_are_overridden() -> None:
    overridden = set(GameLengthAssociationMetric.__dict__)

    assert {"_values", "_extra_accumulate", "_label"} <= overridden
    assert not {"accumulate", "finalize"} & overridden


def test_writes_one_row_per_card_with_the_expected_columns(tmp_path: Path) -> None:
    binder = binder_with_cards([OWLBEAR])
    metric = GameLengthAssociationMetric(VERSION, output_path=tmp_path / "out.parquet")

    df = scan_into_frame(tmp_path, binder, [row(won=True, owlbear_deck=1)], metric)

    assert list(df.reset_index().columns) == [
        "nocab_uuid",
        "game_length_association",
        "sample_count",
    ]
    assert len(df) == 1
