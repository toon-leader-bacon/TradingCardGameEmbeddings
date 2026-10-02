"""OnPlayWinCounts - one subject's games and wins split by on_play, and
the on-play/on-draw win rate delta they give
(see game_data/README.md).

Shared by OnPlayWinRateDeltaMetric (the subject is a card in the deck)
and OnPlayWinRateSensitivityByDeckMetric (the subject is a whole deck):
both tally the same four counts and write the same nullable delta. The
two metrics tally over different keys (deck columns vs. distinct
decks), so only the counts and the delta live here.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# How many tallies row_masks() returns, and from_tallies() reads.
TALLY_COUNT = 4


@dataclass(frozen=True)
class OnPlayWinCounts:
    """Games and wins on each side of on_play, for one card or deck.

    play_games, play_wins: games on the play, and how many were won.
    draw_games, draw_wins: the same, on the draw.
    """

    play_games: int
    play_wins: int
    draw_games: int
    draw_wins: int

    @staticmethod
    def row_masks(
        won: npt.NDArray[np.bool_], on_play: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.bool_]:
        """Which rows count toward each tally, in field order.

        Inputs: won, on_play (shape (rows,)).
        Output: shape (TALLY_COUNT, rows): play_games, play_wins,
            draw_games, draw_wins.
        Side effects: none. Exceptions: none.

        Example:
            >>> masks = OnPlayWinCounts.row_masks(chunk.won, chunk.on_play)
            >>> masks.astype(np.int64) @ deck.present()  # per-column tallies
        """
        on_draw = ~on_play
        return np.stack([on_play, on_play & won, on_draw, on_draw & won])

    @classmethod
    def from_tallies(cls, tallies: npt.NDArray[np.int64]) -> "OnPlayWinCounts":
        """Counts from a (TALLY_COUNT,) tally vector in row_masks()
        order (Factory Method).

        Inputs: tallies. Output: OnPlayWinCounts.
        Side effects: none.
        Exceptions: ValueError if tallies' shape is not (TALLY_COUNT,).

        Example:
            >>> OnPlayWinCounts.from_tallies(np.array([2, 2, 1, 0]))
            OnPlayWinCounts(play_games=2, play_wins=2, draw_games=1, draw_wins=0)
        """
        if tallies.shape != (TALLY_COUNT,):
            raise ValueError(f"expected {TALLY_COUNT} tallies, got {tallies.shape}")
        play_games, play_wins, draw_games, draw_wins = (int(t) for t in tallies)
        return cls(play_games, play_wins, draw_games, draw_wins)

    def game_count(self) -> int:
        """Games on either side.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.

        Example:
            >>> OnPlayWinCounts(2, 2, 1, 0).game_count()
            3
        """
        return self.play_games + self.draw_games

    def win_rate_delta(self) -> float | None:
        """P(won | on_play) - P(won | on_draw), or None when either side
        has no games (undefined, not guessed). Computed as the row
        implementation did: each side's wins / games, then subtracted.

        Inputs: none. Output: float or None.
        Side effects: none. Exceptions: none.

        Example:
            >>> OnPlayWinCounts(2, 2, 1, 0).win_rate_delta()
            1.0
        """
        if self.play_games == 0 or self.draw_games == 0:
            return None
        return self.play_wins / self.play_games - self.draw_wins / self.draw_games
