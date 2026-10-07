"""The two kinds of 17lands metric output a slice can be built from.

Every 17lands metric writes one partition file per source CSV
(partition.py) and is one of two kinds:

- CountTableMetric: rows keyed by KEY_COLUMNS carrying raw, summable
  COUNT_COLUMNS, never a finished rate. A slice sums the counts of its
  partitions per key, then output_from_counts() turns the sums into the
  finished table a dojo reads. Summing then dividing is exact for any
  slice.
- RowStreamMetric: one row per game (or pick, or half-turn). A slice
  filters and concatenates its partitions; nothing is summed.

A metric declares its kind structurally, through the ClassVars below;
is_count_table() is the one place the slice builder tells them apart.

BASELINE ROWS: a count table whose label needs a slice-wide baseline
(GameLengthAssociationMetric: a card's mean turns minus the slice's
mean) sets HAS_BASELINE and writes one extra row per partition file
with every key column null, holding the baseline's counts. The slice
builder sums those rows separately and passes the one-row result to
output_from_counts().
"""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, TypeGuard

import pyarrow as pa

from src.data_retrieval.seventeenlands.refs import DataType


class SlicedMetric(Protocol):
    """What every 17lands metric class declares, whatever its kind.

    FAMILY: the raw export it reads.
    OUTPUT_STEM: its partition directory name and the stem of every
        slice file built from it. Replaces DEFAULT_OUTPUT_PATH: a
        17lands metric has no single output file.
    LABEL_COLUMN: the label column of the finished table a dojo reads.
    """

    FAMILY: ClassVar[DataType]
    OUTPUT_STEM: ClassVar[str]
    LABEL_COLUMN: ClassVar[str]


class CountTableMetric(SlicedMetric, Protocol):
    """A metric whose partitions hold summable counts per key.

    KEY_COLUMNS: e.g. ("nocab_uuid",) or ("deck_uuid",); str columns.
    COUNT_COLUMNS: e.g. ("wins", "games"); int64 or float64 columns.
    HAS_BASELINE: whether each partition carries a null-key baseline row
        (module docstring).
    LABEL_VERSION: bumped whenever output_from_counts() changes, or a
        shared label helper it calls (count_table.ratio_output,
        on_play_win_counts.on_play_delta_output: bump every caller), so slice files built by the old
        logic are rebuilt rather than reused (slice_file.py).
    """

    LABEL_VERSION: ClassVar[int]
    KEY_COLUMNS: ClassVar[tuple[str, ...]]
    COUNT_COLUMNS: ClassVar[tuple[str, ...]]
    HAS_BASELINE: ClassVar[bool]

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """The finished table for one slice, from its summed counts.

        Inputs:
            summed: KEY_COLUMNS + COUNT_COLUMNS, one row per key, summed
                over the slice's partitions (baseline rows excluded).
            baseline: one row of COUNT_COLUMNS summed over the slice's
                baseline rows; None unless HAS_BASELINE.
        Output: by default KEY_COLUMNS + LABEL_COLUMN + "sample_count".
            A metric may instead regroup the keys into the shape its
            dojo reads, as PickNumberDecayCurveMetric does (one row per
            card with per-pick-number lists); its docstring says so. A
            label is null where it is undefined (e.g. no games on one
            side of a delta).
        Side effects: none.
        Exceptions: ValueError if HAS_BASELINE and baseline is None.
        """
        ...


class RowStreamMetric(SlicedMetric, Protocol):
    """A metric whose partitions hold one row per game, pick or
    half-turn; a slice concatenates them.

    IS_ROW_STREAM: marks the kind (always True).
    """

    IS_ROW_STREAM: ClassVar[Literal[True]]


SlicedMetricClass = type[CountTableMetric] | type[RowStreamMetric]


def is_count_table(metric: SlicedMetricClass) -> TypeGuard[type[CountTableMetric]]:
    """Whether metric is a count table (else a row stream).

    Inputs: metric, a 17lands metric class.
    Output: True for a CountTableMetric.
    Side effects: none.
    Exceptions: TypeError if metric declares both kinds, or neither.

    Example:
        >>> is_count_table(WinRateWhenInDeckMetric)
        True
    """
    is_count = hasattr(metric, "COUNT_COLUMNS")
    is_stream = getattr(metric, "IS_ROW_STREAM", False) is True

    # Exactly one kind, never both or neither
    if is_count == is_stream:
        raise TypeError(f"{metric!r} must be exactly one of count table, row stream")
    return is_count
