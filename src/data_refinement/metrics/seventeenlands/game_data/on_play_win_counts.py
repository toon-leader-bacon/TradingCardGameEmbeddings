"""Games and wins split by on_play, and the on-play/on-draw win rate
delta they give (see game_data/README.md).

Shared by OnPlayWinRateDeltaMetric (the subject is a card in the deck)
and OnPlayWinRateSensitivityByDeckMetric (the subject is a whole deck).
Both tally the same four counts (COUNT_COLUMNS) per subject and label a
slice with the same nullable delta. They tally over different keys (deck
columns vs. distinct decks), so only the counts and the delta live here:
this module is the one place the delta rule is written.
"""

import numpy as np
import numpy.typing as npt
import pyarrow as pa

# The four tallies, in on_play_row_masks() order: the count-table
# columns of every metric built on them. Games come first on each side.
COUNT_COLUMNS = ("play_games", "play_wins", "draw_games", "draw_wins")
TALLY_COUNT = len(COUNT_COLUMNS)

_PLAY_GAMES, _PLAY_WINS, _DRAW_GAMES, _DRAW_WINS = range(TALLY_COUNT)


def on_play_row_masks(
    won: npt.NDArray[np.bool_], on_play: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    """Which rows count toward each tally, in COUNT_COLUMNS order.

    Inputs: won, on_play (shape (rows,)).
    Output: shape (TALLY_COUNT, rows): play_games, play_wins,
        draw_games, draw_wins.
    Side effects: none. Exceptions: none.

    Example:
        >>> masks = on_play_row_masks(chunk.won, chunk.on_play)
        >>> masks.astype(np.int64) @ deck.present()  # per-column tallies
    """
    on_draw = ~on_play
    return np.stack([on_play, on_play & won, on_draw, on_draw & won])


def has_on_play_games(tallies: npt.NDArray[np.generic]) -> bool:
    """Whether one subject's tallies hold a game on either side.

    Inputs: tallies, shape (TALLY_COUNT,), in COUNT_COLUMNS order.
    Output: bool.
    Side effects: none. Exceptions: none.

    Example:
        >>> has_on_play_games(np.array([0, 0, 1, 0]))
        True
    """
    return bool(tallies[_PLAY_GAMES] + tallies[_DRAW_GAMES] > 0)


def on_play_delta_output(
    summed: pa.Table, key_columns: tuple[str, ...], label_column: str
) -> pa.Table:
    """The finished table for a count table holding COUNT_COLUMNS:
    P(won | on_play) - P(won | on_draw) per key.

    Inputs: summed (key_columns + COUNT_COLUMNS, summed per key),
        key_columns, label_column.
    Output: key_columns + label_column (float64; null where either side
        has no games, never guessed) + sample_count (play_games +
        draw_games, int64), one row per key with at least one game.
    Side effects: none.
    Exceptions: KeyError if a count column is missing from summed.

    Example:
        >>> on_play_delta_output(summed, ("nocab_uuid",), "on_play_win_rate_delta")
    """
    play_games, play_wins, draw_games, draw_wins = (
        summed.column(name).to_numpy().astype(np.float64) for name in COUNT_COLUMNS
    )
    game_count = play_games + draw_games
    kept = game_count > 0

    # Each side's rate, then the delta; null where a side has no games
    both_sides = (play_games > 0) & (draw_games > 0)
    delta = np.full(len(game_count), np.nan)
    np.divide(play_wins, play_games, out=delta, where=both_sides)
    delta[both_sides] -= draw_wins[both_sides] / draw_games[both_sides]

    result = summed.select(list(key_columns)).filter(pa.array(kept))
    result = result.append_column(
        label_column,
        pa.array(delta[kept], pa.float64(), mask=~both_sides[kept]),
    )
    return result.append_column(
        "sample_count", pa.array(np.rint(game_count[kept]).astype(np.int64))
    )
