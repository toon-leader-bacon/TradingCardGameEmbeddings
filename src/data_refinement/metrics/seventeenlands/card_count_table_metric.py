"""Template Method base for every per-card-column count table of a
17lands chunk family: one CardColumnTallies over one card-column family's
columns, written as nocab_uuid plus COUNT_COLUMNS per card (see
sliced_metric.py, CountTableMetric). Generic over the chunk type.

The base owns every shared step: the tallies, accumulate()'s add,
finalize()'s grouping by card and write. A subclass fixes FAMILY,
OUTPUT_STEM, LABEL_COLUMN and COUNT_COLUMNS, implements
_column_card_uuids(chunk) (which columns are tallied), _increments(chunk)
and output_from_counts(), and may override:

- _has_samples(tallies): whether a card earns a row (default: its first
  count, the denominator by convention, is above zero);
- _extra_accumulate(chunk): per-chunk bookkeeping not gated on any card;
- _baseline_counts(): the partition's baseline row, in COUNT_COLUMNS
  order (default None: no row; see sliced_metric.py's BASELINE ROWS).

Subclasses: game_data's GameCardCountTableMetric (its zone-keyed
counts; game_data/card_count_table_metric.py) and replay_data's
DeckEventRateMetric (replay_data/deck_event_rate_metric.py).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, TypeVar
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.count_table import (
    write_count_table,
)
from src.data_refinement.metrics.seventeenlands.card_column_tallies import (
    CardColumnTallies,
)
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.data_retrieval.seventeenlands.refs import DataType

ChunkT = TypeVar("ChunkT")


class CardCountTableMetric(ABC, Generic[ChunkT]):
    """Per card column: COUNT_COLUMNS tallies, summed over a CSV.

    Satisfies the Metric[ChunkT] Protocol (../metric.py) and
    CountTableMetric (sliced_metric.py) structurally.
    """

    FAMILY: ClassVar[DataType]
    OUTPUT_STEM: ClassVar[str]
    LABEL_COLUMN: ClassVar[str]
    LABEL_VERSION: ClassVar[int] = 1
    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ("nocab_uuid",)
    COUNT_COLUMNS: ClassVar[tuple[str, ...]]
    HAS_BASELINE: ClassVar[bool] = False

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path,
    ) -> None:
        """Start a metric with no tallies; the first chunk sizes them.

        Inputs:
            version_metadata: the CardBinder version this run reads,
                stamped onto the output.
            output_path: this CSV's partition path
                (SeventeenLandsPartition.path()).
        Output: none (constructor).
        Side effects: none (no I/O until finalize()).
        Exceptions: none.
        """
        self._version_metadata = version_metadata
        self._output_path = output_path
        # float64 throughout: exact for any realistic count (< 2**53),
        # and one dtype for every subclass's sums and counts
        self._tallies = CardColumnTallies(
            type(self).__name__, len(self.COUNT_COLUMNS), np.float64
        )

    def accumulate(self, chunk: ChunkT) -> None:
        """Add this chunk's per-column increments, then run the optional
        extra step.

        Inputs: chunk.
        Output: none.
        Side effects: updates the per-column tallies; calls
            _extra_accumulate(chunk) once.
        Exceptions: ValueError if chunk's columns differ from the first
            chunk's (chunks from two CSVs), or _increments()' shape is
            not (len(COUNT_COLUMNS), columns).

        Example:
            >>> metric = TutorTargetRateMetric(version_metadata, path)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        self._tallies.add(self._column_card_uuids(chunk), self._increments(chunk))
        self._extra_accumulate(chunk)

    def finalize(self) -> Path:
        """Group the tallies by card and write this partition's counts.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: writes self._output_path via write_count_table:
            nocab_uuid, then COUNT_COLUMNS (float64), one row per card with
            _has_samples(), plus the baseline row when HAS_BASELINE. A
            CSV with no rows writes the full schema.
        Exceptions: whatever the parquet write raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/tutor_target_rate/KTK/TradDraft.parquet')
        """
        card_uuids, counts = self._tallies.count_columns(
            self.COUNT_COLUMNS, self._has_samples
        )
        return write_count_table(
            self._output_path,
            type(self),
            {"nocab_uuid": card_uuids},
            counts,
            self._version_metadata,
            baseline=self._baseline_counts(),
        )

    @classmethod
    @abstractmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts()."""
        raise NotImplementedError

    @abstractmethod
    def _column_card_uuids(self, chunk: ChunkT) -> tuple[UUID, ...]:
        """The tallied columns' cards, one uuid per column (fixed for a
        CSV).

        Inputs: chunk. Output: tuple of card uuids.
        Side effects: none. Exceptions: none.
        """
        raise NotImplementedError

    @abstractmethod
    def _increments(self, chunk: ChunkT) -> npt.NDArray[np.generic]:
        """This chunk's per-column increments.

        Inputs: chunk.
        Output: shape (len(COUNT_COLUMNS), column count), in
            COUNT_COLUMNS order, castable to float64.
        Side effects: none expected. Exceptions: implementation-defined.
        """
        raise NotImplementedError

    def _has_samples(self, tallies: npt.NDArray[np.generic]) -> bool:
        """Whether a card's summed tallies earn it a row: its first count
        (the denominator, by convention) is above zero.

        Inputs: tallies, shape (len(COUNT_COLUMNS),). Output: bool.
        Side effects: none. Exceptions: none.
        """
        return bool(tallies[0] > 0)

    def _extra_accumulate(self, chunk: ChunkT) -> None:
        """Optional per-chunk bookkeeping beyond the per-card tallies;
        a no-op by default.

        Inputs: chunk. Output: none.
        Side effects: none by default. Exceptions: implementation-defined.
        """
        return

    def _baseline_counts(self) -> npt.NDArray[np.generic] | None:
        """This partition's baseline row in COUNT_COLUMNS order; None
        (no row) by default. Must be non-None exactly when HAS_BASELINE.

        Inputs: none. Output: shape (len(COUNT_COLUMNS),) or None.
        Side effects: none. Exceptions: none.
        """
        return None
