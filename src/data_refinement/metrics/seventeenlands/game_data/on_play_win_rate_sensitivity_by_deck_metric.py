"""OnPlayWinRateSensitivityByDeckMetric - BRAINSTORM.md's multi-card
metric "On-Play Win-Rate Sensitivity by Deck": the deck-level mirror of
on_play_win_rate_delta_metric.py's OnPlayWinRateDeltaMetric - per deck,
P(won | on_play) - P(won | on_draw), aggregated across every game
sharing an identical deck.

A vectorized Metric[GameDataChunk] (see game_data/README.md).
Per chunk it counts each distinct deck's four OnPlayWinCounts tallies
over its rows in one pass, and adds them to a running per-deck_uuid
total - no DeckBox of its own. The canonical DeckBox
(src/data_refinement/deck_box/seventeenlands_game_data/
extraction_stage.py) already holds every deck this metric would ever
see, under the same deck_uuid_from_cards() identity.

ACCUMULATION, NOT STREAMING: unlike game_deck_label_metric.py's
GameDeckLabelMetric family, this metric's label needs every game with
an identical deck, so it does NOT subclass GameDeckLabelMetric.

NULLABLE OUTPUT: same on-play/on-draw-side-must-both-have-samples rule
as OnPlayWinRateDeltaMetric (OnPlayWinCounts.win_rate_delta()) - if a
deck was never seen on one side, its delta is written as None.
"""

import dataclasses
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    ChunkDecks,
    GameDataChunk,
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
    "data/metrics/seventeenlands/game_data/on_play_win_rate_sensitivity_by_deck.parquet"
)


class OnPlayWinRateSensitivityByDeckMetric:
    """Deck -> P(won | on_play) - P(won | on_draw), aggregated across
    every game sharing that exact deck.

    Satisfies the Metric[GameDataChunk] Protocol (../../metric.py)
    structurally.
    """

    LABEL_COLUMN: ClassVar[str] = "on_play_win_rate_sensitivity"
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path | None = None,
    ) -> None:
        """Start a metric with no tallies.

        Inputs:
            version_metadata: the CardBinder version this run reads;
                stamped onto the output with requires_deck_box=True
                (a downstream reader of deck_uuid still needs some
                DeckBox - see module docstring).
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none (no I/O until accumulate()/finalize()).
        Exceptions: none.
        """
        self._version_metadata = dataclasses.replace(
            version_metadata, requires_deck_box=True
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._tallies: dict[UUID, npt.NDArray[np.int64]] = {}

    def accumulate(self, chunk: GameDataChunk) -> None:
        """Add each deck's games and wins on each side of on_play to its
        running total.

        Inputs: chunk.
        Output: none.
        Side effects: updates the per-deck tallies.
        Exceptions: none.

        Example:
            >>> metric = OnPlayWinRateSensitivityByDeckMetric(version_metadata)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        # Each distinct deck's four tallies over this chunk's rows
        masks = OnPlayWinCounts.row_masks(chunk.won, chunk.on_play)
        per_deck = _count_per_deck(masks, chunk.decks)

        # Add them to the running per-deck_uuid totals (a copy, so no
        # stored total keeps this chunk's whole array alive)
        for deck_index, deck in enumerate(chunk.decks.decks):
            previous = self._tallies.get(deck.nocab_uuid)
            increments = per_deck[:, deck_index]
            self._tallies[deck.nocab_uuid] = (
                increments.copy() if previous is None else previous + increments
            )

    def finalize(self) -> Path:
        """Compute every seen deck's on-play/on-draw win rate delta and
        write one row per deck to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns deck_uuid: str, on_play_win_rate_sensitivity: float
            | None, sample_count: int - one row per deck seen at least
            once). Built from row dicts, as the row implementation did.
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/on_play_win_rate_sensitivity_by_deck.parquet')
        """
        result: list[dict] = [
            _sensitivity_row(deck_uuid, OnPlayWinCounts.from_tallies(tallies))
            for deck_uuid, tallies in self._tallies.items()
        ]

        write_dataframe_with_version_metadata(
            pd.DataFrame(result), self._output_path, self._version_metadata
        )
        return self._output_path


def _count_per_deck(
    masks: npt.NDArray[np.bool_], decks: ChunkDecks
) -> npt.NDArray[np.int64]:
    """Per mask and distinct deck, how many of the deck's rows are in the
    mask.

    Inputs: masks (TALLY_COUNT, rows), decks.
    Output: shape (TALLY_COUNT, len(decks.decks)).
    Side effects: none. Exceptions: none.
    """
    assert masks.shape[0] == TALLY_COUNT
    deck_count = len(decks.decks)
    return np.stack(
        [np.bincount(decks.row_deck[mask], minlength=deck_count) for mask in masks]
    ).astype(np.int64)


def _sensitivity_row(deck_uuid: UUID, counts: OnPlayWinCounts) -> dict:
    """One output row: deck_uuid, on_play_win_rate_sensitivity,
    sample_count.

    Inputs: deck_uuid, counts. Output: dict keyed by the output columns.
    Side effects: none. Exceptions: none.
    """
    return {
        "deck_uuid": str(deck_uuid),
        "on_play_win_rate_sensitivity": counts.win_rate_delta(),
        "sample_count": counts.game_count(),
    }
