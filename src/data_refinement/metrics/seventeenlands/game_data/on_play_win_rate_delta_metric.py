"""OnPlayWinRateDeltaMetric - BRAINSTORM.md's single-card metric
"On-Play vs. On-Draw Win Rate Delta": per card, P(won | card in deck,
on_play=True) - P(won | card in deck, on_play=False) - a tempo/curve-
sensitivity proxy.

A GameCardCountTableMetric (card_count_table_metric.py): each partition
holds, per card in the deck, the four on_play_win_counts COUNT_COLUMNS;
the delta is taken only when a slice is built.

NULLABLE OUTPUT: if a card was never seen on one side (on_play or
on_draw) in a slice, that side's rate is undefined - the card's delta is
null rather than guessed at (on_play_win_counts.on_play_delta_output).
"""

from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.game_data.card_count_table_metric import (
    GameCardCountTableMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_counts import (
    COUNT_COLUMNS,
    has_on_play_games,
    on_play_delta_output,
    on_play_row_masks,
)


class OnPlayWinRateDeltaMetric(GameCardCountTableMetric):
    """Card -> P(won | in deck, on_play) - P(won | in deck, on_draw)."""

    OUTPUT_STEM: ClassVar[str] = "on_play_win_rate_delta"
    LABEL_COLUMN: ClassVar[str] = "on_play_win_rate_delta"
    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = COUNT_COLUMNS
    ZONE: ClassVar[GameZone] = GameZone.DECK

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): the on-play minus
        on-draw win rate per card (null when either side has no games).

        Inputs: summed (nocab_uuid + the four counts), baseline (None).
        Output: nocab_uuid, on_play_win_rate_delta, sample_count.
        Side effects: none. Exceptions: none.

        Example:
            >>> OnPlayWinRateDeltaMetric.output_from_counts(summed, None)
        """
        return on_play_delta_output(summed, cls.KEY_COLUMNS, cls.LABEL_COLUMN)

    def _increments(self, chunk: GameDataChunk) -> npt.NDArray[np.float64]:
        """See CardCountTableMetric._increments(): per deck column, each
        on-play tally over the rows its card is present in.

        Inputs: chunk.
        Output: shape (4, deck columns), in COUNT_COLUMNS order.
        Side effects: none. Exceptions: none.
        """
        masks = on_play_row_masks(chunk.won, chunk.on_play)
        present = chunk.zones[GameZone.DECK].present()

        # float64 matmul is BLAS-fast and exact for any chunk's counts
        return masks.astype(np.float64) @ present.astype(np.float64)

    def _has_samples(self, tallies: npt.NDArray[np.generic]) -> bool:
        """A card earns a row with a game on either side (the first
        count alone is on-play games only).

        Inputs: tallies, shape (4,). Output: bool.
        Side effects: none. Exceptions: none.
        """
        return has_on_play_games(tallies)
