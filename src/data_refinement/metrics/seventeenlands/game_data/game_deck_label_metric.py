"""Template Method base for streaming metrics whose training input is
one game's constructed deck (deck_<name>, referenced by deck_uuid,
never embedded - mirroring sts_gg/deck_label_metric.py's own DeckBox
convention) paired with a single scalar label from that same game.

A vectorized Metric[GameDataChunk] (see game_data/README.md).
Each chunk already carries every row's deck (ChunkDecks,
identified once by the parser), so accumulate() writes one output row
per game for the whole chunk at once, using chunk.decks.row_deck_uuids()
for each row's deck_uuid - no DeckBox of its own. The ids use the
canonical DeckBox's identity (deck_uuid_from_cards(), the same
multiset hash as src/data_refinement/deck_box/seventeenlands_game_data/
extraction_stage.py).

KNOWN GAP (2026-10-06): the canonical box stores one deck per draft
(its canonical game), while these deck ids are per game, so a game
played with another build or after sideboarding has no box entry (about
half of KTK.TradSealed's game rows); see game_data/README.md.

Three of round 1's multi-card metrics (game_deck_label_metrics.py's
DeckWinPredictionMetric, DeckGameLengthPredictionMetric,
DeckRankTierPredictionMetric) share every step and differ only in which
per-row value is the label and what type/column name it is written
under. That's a Template Method (PATTERNS.md): this class owns every
shared step; a subclass only fixes LABEL_COLUMN/LABEL_TYPE/
OUTPUT_STEM and implements _labels().

A RowStreamMetric (../sliced_metric.py): a slice concatenates its
partitions.

on_play_win_rate_sensitivity_by_deck_metric.py's
OnPlayWinRateSensitivityByDeckMetric is accumulation, not streaming
(its label needs every game sharing a deck), so it does not subclass
this class.

requires_deck_box=True is still stamped on the output schema: a deck_uuid
output column is only useful to a downstream reader (the training-side
dojo, src/dojos/generic/generic_dojo.py) that can resolve it back to
card lists via SOME DeckBox - the canonical one. This metric itself no
longer depends on any box.

PER-GAME IDENTIFIER: game_data has no single unique-id column - this
container's settled composite is (draft_id: str, match_number: int,
game_number: int), written as three separate output columns, mirroring
draft_data's own draft_id/pack_number/pick_number convention rather
than one joined string.
"""

import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.data_retrieval.seventeenlands.refs import DataType


class GameDeckLabelMetric(ABC):
    """One game's constructed deck -> (deck_uuid, label), one row per
    game, written as soon as accumulate() sees it.

    Satisfies the Metric[GameDataChunk] Protocol (../../metric.py) and
    RowStreamMetric (../sliced_metric.py) structurally.
    """

    LABEL_COLUMN: ClassVar[str]
    LABEL_TYPE: ClassVar[pa.DataType]
    FAMILY: ClassVar[DataType] = DataType.GAME
    OUTPUT_STEM: ClassVar[str]
    IS_ROW_STREAM: ClassVar[Literal[True]] = True

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path,
    ) -> None:
        """Open the output for streaming.

        Inputs:
            version_metadata: the CardBinder version this run reads;
                stamped onto the output with requires_deck_box=True (a
                downstream reader of deck_uuid still needs the canonical
                DeckBox - see module docstring).
            output_path: this CSV's partition path
                (SeventeenLandsPartition.path()).
        Output: none (constructor).
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file) via a ParquetBuilder held open for the
            lifetime of this instance - callers MUST call finalize()
            when done, or the file is left incomplete.
        Exceptions: whatever ParquetBuilder raises on failure to open
            output_path for writing.
        """
        self._output_path = output_path
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
        """Write one output row per game.

        Inputs: chunk.
        Output: none.
        Side effects: writes len(chunk) rows - draft_id, match_number,
            game_number, deck_uuid and the label - to the open
            ParquetBuilder.
        Exceptions: implementation-defined by _labels().

        Example:
            >>> metric = DeckWinPredictionMetric(version_metadata, partition_path)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        labels = self._labels(chunk)

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
        a second call is a no-op.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/deck_win_prediction/KTK/TradDraft.parquet')
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
