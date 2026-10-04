"""OnPlayWinRateSensitivityByDeckMetric - BRAINSTORM.md's multi-card
metric "On-Play Win-Rate Sensitivity by Deck": the deck-level mirror of
on_play_win_rate_delta_metric.py's OnPlayWinRateDeltaMetric - per deck,
P(won | on_play) - P(won | on_draw), aggregated across every game
sharing an identical deck.

A vectorized Metric[GameDataChunk] (see game_data/README.md).
Per chunk it stores the chunk's distinct decks in the shared
DeckBox (store_chunk_decks()), counts each distinct deck's four
on_play_win_counts tallies over its rows in one pass, and adds them to a
running per-deck_uuid total. A CountTableMetric (../sliced_metric.py):
each partition holds those four counts per deck_uuid.

ACCUMULATION, NOT STREAMING: unlike game_deck_label_metric.py's
GameDeckLabelMetric family, this metric's label needs every game with
an identical deck, so it does NOT subclass GameDeckLabelMetric.

NULLABLE OUTPUT: same on-play/on-draw-side-must-both-have-samples rule
as OnPlayWinRateDeltaMetric (on_play_win_counts.on_play_delta_output())
- a deck never seen on one side in a slice gets a null delta.
"""

import dataclasses
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.count_table import (
    write_count_table,
)
from src.data_refinement.metrics.seventeenlands.chunk_decks import (
    store_chunk_decks,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
)
from src.data_refinement.metrics.seventeenlands.chunk_decks import (
    ChunkDecks,
)
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_counts import (
    COUNT_COLUMNS,
    TALLY_COUNT,
    on_play_delta_output,
    on_play_row_masks,
)
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.data_retrieval.seventeenlands.refs import DataType


class OnPlayWinRateSensitivityByDeckMetric:
    """Deck -> P(won | on_play) - P(won | on_draw), aggregated across
    every game sharing that exact deck.

    Satisfies the Metric[GameDataChunk] Protocol (../../metric.py) and
    CountTableMetric (../sliced_metric.py) structurally. A deck played
    in two sets or formats sums across them in a slice that spans both.
    """

    FAMILY: ClassVar[DataType] = DataType.GAME
    OUTPUT_STEM: ClassVar[str] = "on_play_win_rate_sensitivity_by_deck"
    LABEL_COLUMN: ClassVar[str] = "on_play_win_rate_sensitivity"
    LABEL_VERSION: ClassVar[int] = 1
    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ("deck_uuid",)
    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = COUNT_COLUMNS
    HAS_BASELINE: ClassVar[bool] = False

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        deck_box: DeckBox,
        output_path: Path,
    ) -> None:
        """Start a metric with no tallies.

        Inputs:
            version_metadata: the CardBinder version this run reads;
                stamped onto the output with requires_deck_box=True.
            deck_box: the metrics-private DeckBox every deck this
                metric sees is written into - shared with every other
                deck-input metric in the same scan pass. Never the
                published deck_box/ box.
            output_path: this CSV's partition path
                (SeventeenLandsPartition.path()).
        Output: none (constructor).
        Side effects: none (no I/O until accumulate()/finalize()).
        Exceptions: none.
        """
        self._deck_box = deck_box
        self._version_metadata = dataclasses.replace(
            version_metadata, requires_deck_box=True
        )
        self._output_path = output_path
        self._tallies: dict[UUID, npt.NDArray[np.int64]] = {}

    def accumulate(self, chunk: GameDataChunk) -> None:
        """Store the chunk's decks and add each deck's games and wins on
        each side of on_play to its running total.

        Inputs: chunk.
        Output: none.
        Side effects: store_chunk_decks(chunk.decks, self._deck_box); updates the
            per-deck tallies.
        Exceptions: DeckBox's batch-wide errors.

        Example:
            >>> metric = OnPlayWinRateSensitivityByDeckMetric(
            ...     version_metadata, deck_box, partition_path
            ... )
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        store_chunk_decks(chunk.decks, self._deck_box)

        # Each distinct deck's four tallies over this chunk's rows
        masks = on_play_row_masks(chunk.won, chunk.on_play)
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
        """Write this partition's per-deck counts.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: writes self._output_path via write_count_table:
            deck_uuid, then on_play_win_counts' four COUNT_COLUMNS (int64),
            one row per deck seen.
        Exceptions: whatever the parquet write raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/on_play_win_rate_sensitivity_by_deck/KTK/TradDraft.parquet')
        """
        deck_uuids = [str(deck_uuid) for deck_uuid in self._tallies]
        return write_count_table(
            self._output_path,
            type(self),
            {"deck_uuid": deck_uuids},
            _count_columns(list(self._tallies.values())),
            self._version_metadata,
        )

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): the on-play minus
        on-draw win rate per deck (null when either side has no games).

        Inputs: summed (deck_uuid + the four counts), baseline (None).
        Output: deck_uuid, on_play_win_rate_sensitivity, sample_count.
        Side effects: none. Exceptions: none.
        """
        return on_play_delta_output(summed, cls.KEY_COLUMNS, cls.LABEL_COLUMN)


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


def _count_columns(
    tallies: list[npt.NDArray[np.int64]],
) -> dict[str, npt.NDArray[np.int64]]:
    """Per-subject tally vectors as one int64 array per COUNT_COLUMNS
    name.

    Inputs: tallies, each shape (TALLY_COUNT,). Output: dict of arrays,
        each len(tallies) long (empty arrays for no tallies).
    Side effects: none. Exceptions: none.
    """
    stacked = (
        np.stack(tallies, axis=1)
        if tallies
        else np.zeros((TALLY_COUNT, 0), dtype=np.int64)
    )
    return {name: stacked[index] for index, name in enumerate(COUNT_COLUMNS)}
