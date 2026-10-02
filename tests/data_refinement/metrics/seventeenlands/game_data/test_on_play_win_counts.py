"""Tests for on_play_win_counts.py's OnPlayWinCounts."""

import numpy as np
import pytest

from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_counts import (
    OnPlayWinCounts,
)


def test_row_masks_are_in_field_order() -> None:
    won = np.array([True, False, True, False])
    on_play = np.array([True, True, False, False])

    masks = OnPlayWinCounts.row_masks(won, on_play)

    counts = OnPlayWinCounts.from_tallies(masks.sum(axis=1))
    assert counts == OnPlayWinCounts(
        play_games=2, play_wins=1, draw_games=2, draw_wins=1
    )


def test_delta_is_play_rate_minus_draw_rate() -> None:
    assert OnPlayWinCounts(4, 3, 2, 1).win_rate_delta() == 3 / 4 - 1 / 2


@pytest.mark.parametrize(
    "counts", [OnPlayWinCounts(0, 0, 2, 1), OnPlayWinCounts(2, 1, 0, 0)]
)
def test_delta_is_none_without_games_on_one_side(counts: OnPlayWinCounts) -> None:
    assert counts.win_rate_delta() is None


def test_game_count_sums_both_sides() -> None:
    assert OnPlayWinCounts(2, 2, 1, 0).game_count() == 3


def test_from_tallies_rejects_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected 4 tallies"):
        OnPlayWinCounts.from_tallies(np.array([1, 2, 3]))
