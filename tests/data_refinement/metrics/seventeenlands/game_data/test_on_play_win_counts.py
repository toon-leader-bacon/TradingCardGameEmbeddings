"""Tests for on_play_win_counts.py: the four on-play tallies and the
delta labels they give."""

import numpy as np
import pyarrow as pa
import pytest

from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_counts import (
    COUNT_COLUMNS,
    has_on_play_games,
    on_play_delta_output,
    on_play_row_masks,
)


def _summed(*counts: tuple[float, float, float, float]) -> pa.Table:
    """A summed count table: one key per counts tuple ("k0", "k1", ...)."""
    columns = {"key": [f"k{i}" for i in range(len(counts))]}
    for index, name in enumerate(COUNT_COLUMNS):
        columns[name] = [float(c[index]) for c in counts]
    return pa.table(columns)


def test_row_masks_are_in_count_column_order() -> None:
    won = np.array([True, False, True, False])
    on_play = np.array([True, True, False, False])

    masks = on_play_row_masks(won, on_play)

    assert dict(zip(COUNT_COLUMNS, masks.sum(axis=1))) == {
        "play_games": 2,
        "play_wins": 1,
        "draw_games": 2,
        "draw_wins": 1,
    }


@pytest.mark.parametrize(
    "tallies, expected",
    [([1, 0, 0, 0], True), ([0, 0, 1, 0], True), ([0, 0, 0, 0], False)],
)
def test_has_games_on_either_side(tallies: list[int], expected: bool) -> None:
    assert has_on_play_games(np.array(tallies)) is expected


def test_delta_is_play_rate_minus_draw_rate() -> None:
    result = on_play_delta_output(_summed((4, 3, 2, 1)), ("key",), "delta")

    assert result.column("delta").to_pylist() == [3 / 4 - 1 / 2]
    assert result.column("sample_count").to_pylist() == [6]


def test_delta_is_null_without_games_on_one_side() -> None:
    result = on_play_delta_output(
        _summed((0, 0, 2, 1), (2, 1, 0, 0)), ("key",), "delta"
    )

    assert result.column("delta").to_pylist() == [None, None]
    assert result.column("sample_count").to_pylist() == [2, 2]


def test_a_key_with_no_games_has_no_row() -> None:
    result = on_play_delta_output(_summed((0, 0, 0, 0), (1, 1, 1, 0)), ("key",), "d")

    assert result.column("key").to_pylist() == ["k1"]
    assert result.column_names == ["key", "d", "sample_count"]
