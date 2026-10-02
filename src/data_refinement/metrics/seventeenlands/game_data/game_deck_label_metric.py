"""Template Method base for streaming metrics whose training input is
one game's constructed deck (deck_<name>, referenced by deck_uuid,
never embedded - mirroring sts_gg/deck_label_metric.py's own DeckBox
convention) paired with a single scalar label from that same game - see
plans/game_data_metrics.md's Component overview #7.

A vectorized Metric[GameDataChunk] (plans/seventeenlands_chunk_scan.md,
slice 2). Each chunk already carries every row's deck (ChunkDecks,
identified once by the parser), so accumulate() stores the chunk's
distinct decks in the shared DeckBox and writes one output row per game
for the whole chunk at once.

Three of round 1's multi-card metrics (game_deck_label_metrics.py's
DeckWinPredictionMetric, DeckGameLengthPredictionMetric,
DeckRankTierPredictionMetric) share every step and differ only in which
per-row value is the label and what type/column name it is written
under. That's a Template Method (PATTERNS.md): this class owns every
shared step; a subclass only fixes LABEL_COLUMN/LABEL_TYPE/
DEFAULT_OUTPUT_PATH and implements _labels().

on_play_win_rate_sensitivity_by_deck_metric.py's
OnPlayWinRateSensitivityByDeckMetric also stores the chunk's decks, but
is accumulation, not streaming (its label needs every game sharing a
deck), so it does not subclass this class; the step they share,
storing the decks, is chunk_decks.store_chunk_decks().

deck_box IS REQUIRED (no default) here: every subclass genuinely needs
one, and every metric in this container that has no use for a deck box
simply doesn't declare the parameter at all.

PER-GAME IDENTIFIER: game_data has no single unique-id column - this
container's settled composite is (draft_id: str, match_number: int,
game_number: int), written as three separate output columns, mirroring
draft_data's own draft_id/pack_number/pick_number convention rather
than one joined string.
"""

import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.seventeenlands.game_data.chunk_decks import (
    store_chunk_decks,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)


class GameDeckLabelMetric(ABC):
    """One game's constructed deck -> (deck_uuid, label), one row per
    game, written as soon as accumulate() sees it.

    Satisfies the Metric[GameDataChunk] Protocol (../../metric.py)
    structurally.
    """

    LABEL_COLUMN: ClassVar[str]
    LABEL_TYPE: ClassVar[pa.DataType]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """Open the output for streaming.

        Inputs:
            version_metadata: the CardBinder version this run reads;
                stamped onto the output with requires_deck_box=True.
            deck_box: the metrics-private DeckBox every deck this
                metric sees is written into - shared with every other
                deck-input metric in the same scan pass, so identical
                decks dedupe against each other. Never the published
                deck_box/ box.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file) via a ParquetBuilder held open for the
            lifetime of this instance - callers MUST call finalize()
            when done, or the file is left incomplete.
        Exceptions: whatever ParquetBuilder raises on failure to open
            output_path for writing.
        """
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        output_schema = schema_with_version_metadata(
            pa.schema(
                [
                    ("draft_id", pa.string()),
                    ("match_number", pa.int64()),
                    ("game_number", pa.int64()),
                    ("deck_uuid", pa.string()),
                    (self.LABEL_COLUMN, self.LABEL_TYPE),
                ]
            ),
            dataclasses.replace(version_metadata, requires_deck_box=True),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(self._output_path, output_schema)

    def accumulate(self, chunk: GameDataChunk) -> None:
        """Store the chunk's decks and write one output row per game.

        Inputs: chunk.
        Output: none.
        Side effects: store_chunk_decks(chunk.decks, self._deck_box) (a deck
            already in the box, from this or another metric sharing it,
            is left as stored); writes len(chunk) rows - draft_id,
            match_number, game_number, deck_uuid and the label - to the
            open ParquetBuilder.
        Exceptions: implementation-defined by _labels(); DeckBox's
            batch-wide errors.

        Example:
            >>> metric = DeckWinPredictionMetric(version_metadata, deck_box)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        labels = self._labels(chunk)

        # Every distinct deck once, then every game's row at once
        store_chunk_decks(chunk.decks, self._deck_box)
        self._writer.write_columns(
            {
                "draft_id": chunk.keys.draft_id,
                "match_number": chunk.keys.match_number,
                "game_number": chunk.keys.game_number,
                "deck_uuid": chunk.decks.row_deck_uuids(),
                self.LABEL_COLUMN: labels,
            }
        )

    def finalize(self) -> Path:
        """Flush any buffered rows and close the underlying writer.

        A true no-op relative to data - every row this instance will
        ever write was already handed over by accumulate(). Idempotent:
        a second call is a no-op. Does NOT save self._deck_box - that's
        the calling driver's own responsibility, since the box is
        shared across metrics and only the driver knows when every
        metric sharing it is done.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/some_label.parquet')
        """
        self._writer.close()
        return self._output_path

    @abstractmethod
    def _labels(self, chunk: GameDataChunk) -> npt.NDArray[np.generic]:
        """This metric's label for every row of chunk.

        The only step of accumulate()'s sequence a subclass overrides.

        Inputs: chunk.
        Output: shape (len(chunk),), values matching LABEL_TYPE.
        Side effects: none expected.
        Exceptions: implementation-defined.
        """
        raise NotImplementedError
