"""Template Method base for accumulation metrics that tally, per card
present (count > 0) in one game_data zone, the running average of one
per-game scalar (plans/seventeenlands_chunk_scan.md, slice 1).

Satisfies Metric[GameDataChunk]: accumulate() takes a whole chunk and
tallies it with array operations, never a per-row Python loop. The base
owns every shared step (tally, group by card, write); a subclass fixes
LABEL_COLUMN / DEFAULT_OUTPUT_PATH / ZONE and implements _values(), and
may override two optional steps:

- _extra_accumulate(chunk): extra per-chunk bookkeeping not gated on
  any card (GameLengthAssociationMetric's format-wide turn baseline);
- _label(value_sum, count): how one card's tallies become its label
  (default: the plain average; GameLengthAssociationMetric subtracts
  its baseline).

Tallies (value sum, count) are kept per matched column in a
CardColumnTallies (card_column_tallies.py) and grouped by card uuid
only in finalize(). Two header columns naming one card therefore still
count twice, exactly as the row implementation did. The count is kept
as a float64 alongside the sum (exact for any realistic game count) and
written as an int.

VERSION METADATA, NOT A BINDER: the driver computes the CardBinder
version once per family run and passes it in; card matching lives in
GameDataChunkParser, so this class never needs the binder or the header.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

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
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)


class GameCardAverageMetric(ABC):
    """Per-card running average of one per-game scalar, across every
    game the card was present in, in this subclass's ZONE.
    """

    LABEL_COLUMN: ClassVar[str]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]
    ZONE: ClassVar[GameZone]

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path | None = None,
    ) -> None:
        """Start a metric with no tallies; the first chunk sizes them.

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
        # Per-column (value_sum, count), sized by the first chunk
        self._tallies = CardColumnTallies(type(self).__name__, 2, np.float64)

    def accumulate(self, chunk: GameDataChunk) -> None:
        """Tally every present (row, column) of this ZONE toward its
        column's (value_sum, count), then run the optional extra step.

        Inputs: chunk.
        Output: none.
        Side effects: updates this metric's per-column tallies; calls
            _extra_accumulate(chunk) once.
        Exceptions: ValueError if chunk's ZONE columns differ from the
            first chunk's (a driver bug: chunks from two CSVs).

        Example:
            >>> metric = WinRateWhenInDeckMetric(version_metadata)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        zone = chunk.zones[self.ZONE]

        # Tally this chunk in one pass over its present matrix
        self._tallies.add(zone.card_uuids, self._increments(zone.present(), chunk))

        # Optional subclass bookkeeping not gated on any card
        self._extra_accumulate(chunk)

    def finalize(self) -> Path:
        """Group the per-column tallies by card and write one row per
        card seen at least once.

        Inputs: none.
        Output: self._output_path.
        Side effects: writes self._output_path (parquet: nocab_uuid
            str, LABEL_COLUMN float, sample_count int, with version
            metadata), creating parent directories. A run with no rows
            writes a zero-row file with that full schema (the row
            implementation wrote one with no columns).
        Exceptions: whatever the parquet write raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/win_rate_when_in_deck.parquet')
        """
        result: list[dict] = []

        # One output row per card with at least one sample
        for card_uuid, (value_sum, count) in self._tallies.per_card().items():
            if int(count) == 0:
                continue
            result.append(
                {
                    "nocab_uuid": str(card_uuid),
                    self.LABEL_COLUMN: self._label(float(value_sum), int(count)),
                    "sample_count": int(count),
                }
            )

        write_dataframe_with_version_metadata(
            self._build_output_frame(result), self._output_path, self._version_metadata
        )
        return self._output_path

    @abstractmethod
    def _values(self, chunk: GameDataChunk) -> npt.NDArray[np.float64]:
        """The scalar to average, one per row of chunk.

        Inputs: chunk. Output: shape (len(chunk),).
        Side effects: none expected. Exceptions: implementation-defined.
        """
        raise NotImplementedError

    def _extra_accumulate(self, chunk: GameDataChunk) -> None:
        """Optional per-chunk bookkeeping beyond the per-card tally.

        No-op by default (see module docstring).

        Inputs: chunk. Output: none.
        Side effects: none by default. Exceptions: implementation-defined.
        """
        return

    def _label(self, value_sum: float, count: int) -> float:
        """One card's label from its tallies: value_sum / count by
        default.

        Inputs: value_sum, count (count >= 1). Output: float.
        Side effects: none. Exceptions: none.
        """
        return value_sum / count

    def _increments(
        self, present: npt.NDArray[np.bool_], chunk: GameDataChunk
    ) -> npt.NDArray[np.float64]:
        """One chunk's per-column (value_sum, count) increments.

        Inputs: present (rows x columns, this ZONE), chunk.
        Output: shape (2, columns): present^T @ _values(chunk), and
            present summed over rows.
        Side effects: none.
        Exceptions: ValueError if _values()' length differs from the
            chunk's row count.
        """
        values = self._values(chunk)
        if values.shape[0] != present.shape[0]:
            raise ValueError(
                f"{type(self).__name__}: {values.shape[0]} values for "
                f"{present.shape[0]} rows"
            )
        return np.stack(
            [
                present.T.astype(np.float64) @ values,
                present.sum(axis=0, dtype=np.float64),
            ]
        )

    def _build_output_frame(self, rows: list[dict]) -> pd.DataFrame:
        """rows as a DataFrame with the full output schema, even when
        empty.

        Inputs: rows. Output: DataFrame with columns nocab_uuid,
            LABEL_COLUMN, sample_count.
        Side effects: none. Exceptions: none.
        """
        frame = pd.DataFrame(
            rows, columns=["nocab_uuid", self.LABEL_COLUMN, "sample_count"]
        )
        return frame.astype(
            {"nocab_uuid": str, self.LABEL_COLUMN: "float64", "sample_count": "int64"}
        )
