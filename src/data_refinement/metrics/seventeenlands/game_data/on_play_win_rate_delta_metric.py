"""OnPlayWinRateDeltaMetric - BRAINSTORM.md's single-card metric
"On-Play vs. On-Draw Win Rate Delta": per card, P(won | card in deck,
on_play=True) - P(won | card in deck, on_play=False) - a tempo/curve-
sensitivity proxy.

A vectorized Metric[GameDataChunk] (see game_data/README.md).
Standalone - does NOT subclass GameCardAverageMetric: that
base keeps one running (value_sum, count) per card, while this metric
keeps four counts per card (games and wins on each side of on_play,
OnPlayWinCounts). Its per-column counts live in a CardColumnTallies,
summed per card in finalize().

NULLABLE OUTPUT: if a card was never seen on one side (on_play or
on_draw) across every scanned game, that side's rate is undefined - the
delta for that card is written as None rather than guessed at, the same
convention draft_data/pick_number_decay_curve_metric.py uses for an
unseen pick_number bucket.
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data_refinement.metrics.seventeenlands.game_data.card_column_tallies import (
    CardColumnTallies,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_counts import (
    TALLY_COUNT,
    OnPlayWinCounts,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/on_play_win_rate_delta.parquet"
)


class OnPlayWinRateDeltaMetric:
    """Card -> P(won | in deck, on_play) - P(won | in deck, on_draw).

    Satisfies the Metric[GameDataChunk] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path | None = None,
    ) -> None:
        """Start a metric with no tallies.

        Inputs:
            version_metadata: the CardBinder version this run reads,
                stamped onto the output.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none (no I/O until finalize()).
        Exceptions: none.
        """
        self._version_metadata = version_metadata
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._tallies = CardColumnTallies(type(self).__name__, TALLY_COUNT, np.int64)

    def accumulate(self, chunk: GameDataChunk) -> None:
        """Tally every deck column's games and wins on each side of
        on_play, for the rows where its card is present.

        Inputs: chunk.
        Output: none.
        Side effects: adds this chunk to the per-column tallies.
        Exceptions: ValueError if chunk's deck columns differ from the
            first chunk's (chunks from two CSVs).

        Example:
            >>> metric = OnPlayWinRateDeltaMetric(version_metadata)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        deck = chunk.zones[GameZone.DECK]

        # Each tally's rows, counted per column where the card is present
        masks = OnPlayWinCounts.row_masks(chunk.won, chunk.on_play)
        self._tallies.add(deck.card_uuids, _count_per_column(masks, deck.present()))

    def finalize(self) -> Path:
        """Compute every seen card's on-play/on-draw win rate delta and
        write one row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, on_play_win_rate_delta: float |
            None, sample_count: int - one row per card seen at least
            once on either side). Built from row dicts, as the row
            implementation did, so a run with no cards writes the same
            column-less file.
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/on_play_win_rate_delta.parquet')
        """
        result: list[dict] = []

        # One output row per card seen in at least one game
        for card_uuid, tallies in self._tallies.per_card().items():
            counts = OnPlayWinCounts.from_tallies(tallies)
            if counts.game_count() == 0:
                continue
            result.append(_delta_row(card_uuid, counts))

        write_dataframe_with_version_metadata(
            pd.DataFrame(result), self._output_path, self._version_metadata
        )
        return self._output_path


def _count_per_column(
    masks: npt.NDArray[np.bool_], present: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int64]:
    """Per mask and column, how many rows are in the mask and present.

    Inputs: masks (tallies, rows), present (rows, columns).
    Output: shape (tallies, columns).
    Side effects: none. Exceptions: none.
    """
    # float64 matmul is BLAS-fast and exact for any chunk's counts
    counts = masks.astype(np.float64) @ present.astype(np.float64)
    return np.rint(counts).astype(np.int64)


def _delta_row(card_uuid: UUID, counts: OnPlayWinCounts) -> dict:
    """One output row: nocab_uuid, on_play_win_rate_delta, sample_count.

    Inputs: card_uuid, counts (game_count() >= 1).
    Output: dict keyed by the output columns.
    Side effects: none. Exceptions: none.
    """
    return {
        "nocab_uuid": str(card_uuid),
        "on_play_win_rate_delta": counts.win_rate_delta(),
        "sample_count": counts.game_count(),
    }
