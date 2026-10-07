"""Template Method base for the replay_data count tables tallied from
turn events, per card code (see replay_data/README.md).

A vectorized Metric[ReplayDataChunk] and a CountTableMetric
(../sliced_metric.py): each partition holds COUNT_COLUMNS per card. The
base owns the CodeTallies (code_tallies.py), remembers the latest code
table, and writes the partition; a subclass fixes OUTPUT_STEM,
LABEL_COLUMN and COUNT_COLUMNS, and implements _tally(chunk), which adds
the chunk's entries to the tallies, and output_from_counts().

Subclasses: ReplayTurnEventRateMetric (and its two concretes),
AverageTurnCastMetric, TurnsToGameEndAfterCastMetric. The deck-column
rates (cast, discard, tutor) tally per deck column instead
(deck_event_rate_metric.py).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.count_table import (
    write_count_table,
)
from src.data_refinement.metrics.seventeenlands.replay_data.code_tallies import (
    CodeTallies,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    ReplayDataChunk,
)
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.data_retrieval.seventeenlands.refs import DataType


class CodeCountTableMetric(ABC):
    """Per card code: COUNT_COLUMNS tallies, summed over a CSV.

    Satisfies the Metric[ReplayDataChunk] Protocol (../../metric.py) and
    CountTableMetric (../sliced_metric.py) structurally.
    """

    FAMILY: ClassVar[DataType] = DataType.REPLAY
    OUTPUT_STEM: ClassVar[str]
    LABEL_COLUMN: ClassVar[str]
    LABEL_VERSION: ClassVar[int] = 1
    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ("nocab_uuid",)
    COUNT_COLUMNS: ClassVar[tuple[str, ...]]
    HAS_BASELINE: ClassVar[bool] = False

    def __init__(
        self, version_metadata: MetricVersionMetadata, output_path: Path
    ) -> None:
        """Start a metric with no tallies.

        Inputs:
            version_metadata: the CardBinder version this run reads,
                stamped onto the output.
            output_path: this CSV's partition path.
        Output: none (constructor).
        Side effects: none (no I/O until finalize()).
        Exceptions: none.
        """
        self._version_metadata = version_metadata
        self._output_path = output_path
        self._tallies = CodeTallies(type(self).__name__, len(self.COUNT_COLUMNS))
        self._card_uuids: tuple[UUID, ...] = ()

    def accumulate(self, chunk: ReplayDataChunk) -> None:
        """Tally this chunk's entries and keep its code table.

        Inputs: chunk.
        Output: none.
        Side effects: adds to the tallies; remembers chunk.card_uuids.
        Exceptions: implementation-defined by _tally().

        Example:
            >>> metric = AverageTurnCastMetric(version_metadata, partition_path)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        self._tally(chunk)
        self._card_uuids = chunk.card_uuids

    def finalize(self) -> Path:
        """Write this partition's counts: one row per card whose first
        count is above zero.

        Inputs: none.
        Output: self._output_path.
        Side effects: writes self._output_path via write_count_table:
            nocab_uuid, then COUNT_COLUMNS (float64).
        Exceptions: whatever the parquet write raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/replay_data/average_turn_cast/PIO/TradSealed.parquet')
        """
        card_uuids, counts = self._tallies.count_columns(
            self._card_uuids, self.COUNT_COLUMNS, lambda tallies: tallies[0] > 0
        )
        return write_count_table(
            self._output_path,
            type(self),
            {"nocab_uuid": card_uuids},
            counts,
            self._version_metadata,
        )

    @classmethod
    @abstractmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts()."""
        raise NotImplementedError

    @abstractmethod
    def _tally(self, chunk: ReplayDataChunk) -> None:
        """Add this chunk's entries to self._tallies, the first count
        (the denominator, by convention) first.

        Inputs: chunk. Output: none.
        Side effects: adds to self._tallies.
        Exceptions: implementation-defined.
        """
        raise NotImplementedError
