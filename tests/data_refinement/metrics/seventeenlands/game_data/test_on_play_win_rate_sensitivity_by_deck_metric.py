"""Tests for on_play_win_rate_sensitivity_by_deck_metric.py's
OnPlayWinRateSensitivityByDeckMetric, driven through scan_game_csv as a
real run drives it."""

from pathlib import Path

import pandas as pd

from src.data_refinement.deck_ids import deck_uuid_from_cards
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_metric import (  # noqa: E501
    OnPlayWinRateSensitivityByDeckMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.scanner import scan_game_csv
from tests.data_refinement.metrics.seventeenlands.game_data._chunk_fixtures import (
    MORNINGSTAR,
    OWLBEAR,
    VERSION,
    binder_with_cards,
    parser_for,
    read_finished,
    row,
    uuid_for,
    write_csv,
)

_BINDER = binder_with_cards([OWLBEAR, MORNINGSTAR])


def _scan(tmp_path: Path, rows: list[dict], block_size: int = 1 << 20) -> pd.DataFrame:
    """Scan rows through the metric; its output indexed by deck_uuid."""
    metric = OnPlayWinRateSensitivityByDeckMetric(
        VERSION, output_path=tmp_path / "out.parquet"
    )
    csv_path = write_csv(tmp_path / "games.csv", rows)
    scan_game_csv(csv_path, [metric], parser_for(_BINDER), block_size=block_size)
    return read_finished(metric).set_index("deck_uuid")


def test_computes_on_play_rate_minus_on_draw_rate_per_deck(tmp_path: Path) -> None:
    # Same deck in both games - on play: 1/1; on draw: 0/1
    df = _scan(
        tmp_path,
        [
            row(on_play=True, won=True, owlbear_deck=4, game_number=1),
            row(on_play=False, won=False, owlbear_deck=4, game_number=2),
        ],
    )

    # owlbear_deck=4: the full multiset is 4 copies of Owlbear.
    deck_uuid = str(deck_uuid_from_cards([uuid_for(_BINDER, OWLBEAR)] * 4))
    assert list(df.index) == [deck_uuid]
    assert df.loc[deck_uuid, "on_play_win_rate_sensitivity"] == 1.0


def test_sensitivity_is_none_when_deck_never_seen_on_one_side(
    tmp_path: Path,
) -> None:
    df = _scan(tmp_path, [row(on_play=True, won=True, owlbear_deck=4)])

    # A written None reads back as NaN in a float64 column
    assert pd.isna(df.iloc[0]["on_play_win_rate_sensitivity"])


def test_identical_decks_across_games_share_one_deck_uuid(
    tmp_path: Path,
) -> None:
    df = _scan(
        tmp_path,
        [
            row(on_play=True, won=True, owlbear_deck=4, game_number=1),
            row(on_play=False, won=False, owlbear_deck=4, game_number=2),
        ],
    )

    assert len(df) == 1


def test_sample_count_sums_both_sides(tmp_path: Path) -> None:
    df = _scan(
        tmp_path,
        [
            row(on_play=True, won=True, owlbear_deck=4),
            row(on_play=True, won=False, owlbear_deck=4),
            row(on_play=False, won=True, owlbear_deck=4),
        ],
    )

    assert df.iloc[0]["sample_count"] == 3


def test_a_deck_spanning_chunks_tallies_like_one_chunk(tmp_path: Path) -> None:
    rows = [
        row(
            won=i % 3 == 0,
            on_play=i % 2 == 0,
            owlbear_deck=1,
            morningstar_deck=i % 2,
            game_number=i,
        )
        for i in range(60)
    ]

    one = _scan(tmp_path, rows)
    many = _scan(tmp_path, rows, block_size=256)

    assert len(one) == 2
    pd.testing.assert_frame_equal(one.sort_index(), many.sort_index())
