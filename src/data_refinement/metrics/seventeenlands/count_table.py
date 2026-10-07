"""Shared helpers for CountTableMetric (sliced_metric.py): writing one
partition's counts, and the label shapes several count tables share.

Writing: every count-table metric's finalize() hands its per-key counts
to write_count_table(), so the partition schema (KEY_COLUMNS, then
COUNT_COLUMNS, plus the optional null-key baseline row) is built in one
place.

Labels: most count tables' output_from_counts() is a ratio of two
counts (ratio_output), called with the metric's own column names. The
on-play delta is game_data-only and lives in
game_data/on_play_win_counts.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.metrics.seventeenlands.sliced_metric import (
    CountTableMetric,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)

SAMPLE_COUNT_COLUMN = "sample_count"


def write_count_table(
    path: Path,
    metric: type[CountTableMetric],
    keys: Mapping[str, Sequence[object]],
    counts: Mapping[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]],
    version_metadata: MetricVersionMetadata,
    baseline: npt.NDArray[np.generic] | None = None,
) -> Path:
    """Write one count-table partition file.

    Inputs:
        path: the partition path (SeventeenLandsPartition.path()).
        metric: the writing metric; its KEY_COLUMNS, COUNT_COLUMNS and
            HAS_BASELINE are the schema checked here, at write time,
            not first at slice-build time.
        keys: one sequence per KEY_COLUMNS name: str (card or deck
            uuids, rank) or int (pack and pick numbers); an empty str
            column is typed string.
        counts: one array per COUNT_COLUMNS name, each as long as keys'.
        version_metadata: stamped onto the file.
        baseline: the baseline row, in COUNT_COLUMNS order; written as
            one extra row with every key null. Must be given exactly
            when metric.HAS_BASELINE.
    Output: path.
    Side effects: creates path's parent directories; writes path. A
        partition with no keys still writes the full schema (zero rows,
        or only the baseline row).
    Exceptions: ValueError if keys' or counts' names are not the
        metric's KEY_COLUMNS / COUNT_COLUMNS (in order), the columns
        differ in length, or baseline is given without HAS_BASELINE (or
        missing with it).

    Example:
        >>> write_count_table(
        ...     path, TutorTargetRateMetric, {"nocab_uuid": uuids},
        ...     {"in_deck": in_deck, "tutored": tutored}, version_metadata,
        ... )
    """
    # Validate: the metric's declared columns, one value per key
    _check_declared_columns(metric, keys, counts, baseline)
    _check_column_lengths(keys, counts)

    # Build the key and count columns, then the optional baseline row
    table = _count_table(keys, counts)
    if baseline is not None:
        table = pa.concat_tables([table, _baseline_row(table.schema, baseline)])

    # Write with the version metadata
    _write_table_with_version_metadata(table, path, version_metadata)
    return path


def ratio_output(
    summed: pa.Table,
    key_columns: tuple[str, ...],
    numerator: str,
    denominator: str,
    label_column: str,
) -> pa.Table:
    """The finished table for a count table whose label is
    numerator / denominator and whose sample count is the denominator.

    Inputs: summed (key_columns + both count columns), the key and
        count column names, label_column.
    Output: key_columns + label_column (float64) + sample_count
        (int64), one row per key with a nonzero denominator; keys with a
        zero denominator are dropped.
    Side effects: none.
    Exceptions: KeyError if a named column is missing from summed.

    Example:
        >>> ratio_output(summed, ("nocab_uuid",), "tutored", "in_deck", "tutor_target_rate")
    """
    numerators = summed.column(numerator).to_numpy().astype(np.float64)
    denominators = summed.column(denominator).to_numpy().astype(np.float64)
    kept = denominators > 0

    # Keys with a sample, their rate, and the denominator as the count
    result = summed.select(list(key_columns)).filter(pa.array(kept))
    result = result.append_column(
        label_column, pa.array(numerators[kept] / denominators[kept], pa.float64())
    )
    return result.append_column(
        SAMPLE_COUNT_COLUMN,
        pa.array(np.rint(denominators[kept]).astype(np.int64), pa.int64()),
    )


def _check_column_lengths(
    keys: Mapping[str, Sequence[object]],
    counts: Mapping[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]],
) -> None:
    """Raise unless every key and count column has the same length.

    Inputs: keys, counts. Output: none.
    Side effects: none. Exceptions: ValueError naming the first
        mismatched column.
    """
    lengths = {name: len(values) for name, values in {**keys, **counts}.items()}
    expected = next(iter(lengths.values()), 0)
    for name, length in lengths.items():
        if length != expected:
            raise ValueError(f"column {name!r} has {length} rows, expected {expected}")


def _count_table(
    keys: Mapping[str, Sequence[object]],
    counts: Mapping[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]],
) -> pa.Table:
    """keys then counts as one table: each key column typed from its
    values (string or int64; an empty one is null-typed, which
    concatenation promotes to its neighbours' type), counts keeping
    their dtype.

    Inputs: keys, counts. Output: pa.Table.
    Side effects: none. Exceptions: none.
    """
    columns = {
        name: pa.array(list(values)) if len(values) else pa.nulls(0)
        for name, values in keys.items()
    }
    columns.update({name: pa.array(values) for name, values in counts.items()})
    return pa.table(columns)


def _check_declared_columns(
    metric: type[CountTableMetric],
    keys: Mapping[str, Sequence[object]],
    counts: Mapping[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]],
    baseline: npt.NDArray[np.generic] | None,
) -> None:
    """Raise unless keys and counts carry exactly the metric's
    KEY_COLUMNS and COUNT_COLUMNS, in order, and baseline is given
    exactly when HAS_BASELINE (with one value per count column).

    Inputs: metric, keys, counts, baseline. Output: none.
    Side effects: none. Exceptions: ValueError naming the mismatch.
    """
    name = metric.__name__
    if tuple(keys) != metric.KEY_COLUMNS:
        raise ValueError(f"{name}: key columns {tuple(keys)} != {metric.KEY_COLUMNS}")
    if tuple(counts) != metric.COUNT_COLUMNS:
        raise ValueError(
            f"{name}: count columns {tuple(counts)} != {metric.COUNT_COLUMNS}"
        )
    if (baseline is not None) != metric.HAS_BASELINE:
        raise ValueError(f"{name}: baseline given must match HAS_BASELINE")
    if baseline is not None and baseline.shape != (len(metric.COUNT_COLUMNS),):
        raise ValueError(f"{name}: baseline shape {baseline.shape} != one per count")


def _baseline_row(schema: pa.Schema, baseline: npt.NDArray[np.generic]) -> pa.Table:
    """One row of schema: every key column null, each count column
    baseline's value (COUNT_COLUMNS order) cast to that column's type.

    Inputs: schema, baseline. Output: a one-row pa.Table.
    Side effects: none. Exceptions: none (checked by
        _check_declared_columns).
    """
    key_count = len(schema) - len(baseline)
    columns = [pa.nulls(1, field.type) for field in list(schema)[:key_count]]
    for field, value in zip(list(schema)[key_count:], baseline):
        columns.append(pa.array([value]).cast(field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def _write_table_with_version_metadata(
    table: pa.Table, path: Path, version_metadata: MetricVersionMetadata
) -> None:
    """Write table to path with version_metadata in its schema.

    Inputs: table, path, version_metadata. Output: none.
    Side effects: creates path's parent directories; writes path.
    Exceptions: whatever pyarrow.parquet.write_table raises.
    """
    schema = schema_with_version_metadata(table.schema, version_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.cast(schema), path)
