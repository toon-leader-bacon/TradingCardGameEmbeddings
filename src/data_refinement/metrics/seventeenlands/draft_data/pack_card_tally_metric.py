"""Template Method base for the draft_data take-rate count tables: per
(card, stratum key), how often the card was in a pack and how often it
was taken (see draft_data/README.md).

A vectorized Metric[DraftDataChunk] and a CountTableMetric
(../sliced_metric.py): each partition holds (in_pack, picked) per
KEY_COLUMNS; the take rate (picked / in_pack) is computed only when a
slice is built.

KEY_COLUMNS is the single source of a subclass's stratification:
"nocab_uuid" then the DraftStratum column names (e.g. "pack_number",
"pick_number"). A subclass fixes OUTPUT_STEM and KEY_COLUMNS and may
override:

- _eligible_rows(chunk): which rows count (default: every row);
- _stratum_values(chunk, stratum): a stratum's per-row values (default:
  chunk.stratum(stratum); PickNumberDecayCurveMetric clamps).

PickNumberDecayCurveMetric (pick_number_decay_curve_metric.py) also
overrides output_from_counts to regroup into per-card lists.

PICKTWO: a PickTwo row's second pick counts as picked too
(DraftDataChunk.picked()), so both cards taken are counted.
"""

from abc import ABC
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.count_table import (
    ratio_output,
    write_count_table,
)
from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk import (
    DraftDataChunk,
    DraftStratum,
)
from src.data_refinement.metrics.seventeenlands.draft_data.keyed_card_tallies import (
    KeyedCardTallies,
)
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.data_retrieval.seventeenlands.refs import DataType

IN_PACK_COLUMN = "in_pack"
PICKED_COLUMN = "picked"


class PackCardTallyMetric(ABC):
    """Per-(card, stratum key) (in_pack, picked) counts.

    Satisfies the Metric[DraftDataChunk] Protocol (../../metric.py) and
    CountTableMetric (../sliced_metric.py) structurally.
    """

    FAMILY: ClassVar[DataType] = DataType.DRAFT
    OUTPUT_STEM: ClassVar[str]
    LABEL_COLUMN: ClassVar[str] = "take_rate"
    LABEL_VERSION: ClassVar[int] = 1
    KEY_COLUMNS: ClassVar[tuple[str, ...]]
    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = (IN_PACK_COLUMN, PICKED_COLUMN)
    HAS_BASELINE: ClassVar[bool] = False

    def __init__(
        self, version_metadata: MetricVersionMetadata, output_path: Path
    ) -> None:
        """Start a metric with no tallies.

        Inputs:
            version_metadata: the CardBinder version this run reads,
                stamped onto the output.
            output_path: this CSV's partition path
                (SeventeenLandsPartition.path()).
        Output: none (constructor).
        Side effects: none (no I/O until finalize()).
        Exceptions: ValueError if KEY_COLUMNS[1:] names no DraftStratum.
        """
        self._version_metadata = version_metadata
        self._output_path = output_path
        self._strata = tuple(DraftStratum(name) for name in self.KEY_COLUMNS[1:])
        self._tallies = KeyedCardTallies(type(self).__name__, len(self.COUNT_COLUMNS))

    def accumulate(self, chunk: DraftDataChunk) -> None:
        """Tally every pack option of every eligible row toward its
        (column, stratum key)'s (in_pack, picked).

        Inputs: chunk.
        Output: none.
        Side effects: adds to the keyed tallies.
        Exceptions: ValueError if chunk's pack columns differ from the
            first chunk's (chunks from two CSVs).

        Example:
            >>> metric = CardTakeRateMetric(version_metadata, partition_path)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        rows = self._eligible_rows(chunk)

        # The eligible rows' strata, pack cells and taken cells
        strata = tuple(
            self._stratum_values(chunk, stratum)[rows] for stratum in self._strata
        )
        increments = np.stack([chunk.pack.present()[rows], chunk.picked()[rows]])
        self._tallies.add(chunk.pack.card_uuids, strata, increments)

    def finalize(self) -> Path:
        """Write this partition's counts: one row per (card, stratum key)
        seen in a pack at least once.

        Inputs: none.
        Output: self._output_path.
        Side effects: writes self._output_path via write_count_table:
            KEY_COLUMNS, in_pack, picked (float64).
        Exceptions: whatever the parquet write raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/draft_data/card_take_rate/KTK/TradDraft.parquet')
        """
        keys, counts = self._tallies.count_columns(
            self.KEY_COLUMNS, self.COUNT_COLUMNS, lambda tallies: tallies[0] > 0
        )
        return write_count_table(
            self._output_path, type(self), keys, counts, self._version_metadata
        )

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): picked / in_pack
        as take_rate, in_pack as sample_count.

        Inputs: summed (KEY_COLUMNS, in_pack, picked), baseline (None).
        Output: KEY_COLUMNS, take_rate, sample_count.
        Side effects: none. Exceptions: none.

        Example:
            >>> CardTakeRateMetric.output_from_counts(summed, None)
        """
        return ratio_output(
            summed, cls.KEY_COLUMNS, PICKED_COLUMN, IN_PACK_COLUMN, cls.LABEL_COLUMN
        )

    def _eligible_rows(self, chunk: DraftDataChunk) -> npt.NDArray[np.bool_]:
        """Which rows count toward the tallies: every row by default.

        Inputs: chunk. Output: bool array (rows,).
        Side effects: none. Exceptions: none.
        """
        return np.ones(len(chunk), np.bool_)

    def _stratum_values(
        self, chunk: DraftDataChunk, stratum: DraftStratum
    ) -> npt.NDArray[np.generic]:
        """One stratum's per-row values: chunk.stratum(stratum) by
        default.

        Inputs: chunk, stratum. Output: array (rows,).
        Side effects: none. Exceptions: none.
        """
        return chunk.stratum(stratum)
