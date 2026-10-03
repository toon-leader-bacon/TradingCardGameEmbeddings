"""SeventeenLandsSliceFile - builds the one parquet file a dojo trains on
for a (17lands metric, slice) pair.

The built file is an ordinary metric parquet: the dojo hands its path
to FileManagerParquet exactly as it would any other metric output. It
lives at

    <metrics_root>/seventeenlands/<family>/slices/<OUTPUT_STEM>.<slice name>.parquet

so its stem is unique per (metric, slice), and GenericDojo's default
dojo name and split prefix (the file stem) never collide.

- Count table: read every selected partition, set the null-key baseline
  rows aside, sum the counts per key, and write the metric's
  output_from_counts() result.
- Row stream: stream every selected partition into one file in
  pyarrow batches, adding "set" and "format" columns.

Caching: the built file carries the partitions' MetricVersionMetadata
plus a fingerprint of its inputs in its schema metadata: each selected
partition's path, size and modification time, and the metric's
LABEL_VERSION (0 for a row stream, which has no label logic here) and
this module's SLICE_BUILD_VERSION. Bump LABEL_VERSION for a change to a
metric's labels; bump SLICE_BUILD_VERSION for a change to how this
module builds any slice (e.g. the row-stream set/format columns).
build() reuses a file whose fingerprint still matches and rebuilds any
other. Building runs on the first dojo construction that asks for the
slice.

Disk cost: a count-table slice is one row per key. A row-stream slice
is a filtered copy of its partitions; that copy is the price of leaving
FileManagerParquet unchanged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data_refinement.metrics.seventeenlands.data_slice import (
    SeventeenLandsSlice,
)
from src.data_refinement.metrics.seventeenlands.partition import (
    METRICS_ROOT,
    SLICES_DIRECTORY,
    SOURCE_DIRECTORY,
    SeventeenLandsPartition,
    find_partitions,
)
from src.data_refinement.metrics.seventeenlands.sliced_metric import (
    CountTableMetric,
    SlicedMetricClass,
    is_count_table,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    read_version_metadata,
    schema_with_version_metadata,
)

# Bump on any change to how slice files are built; part of every
# fingerprint, so every slice file is rebuilt.
SLICE_BUILD_VERSION = 1
SET_COLUMN = "set"
FORMAT_COLUMN = "format"
_FINGERPRINT_KEY = b"seventeenlands_slice_inputs"


class PartitionStamp(NamedTuple):
    """One partition file as a slice file saw it.

    relative_path: its path relative to the metrics root, POSIX style.
    size_bytes, modified_ns: its os.stat() size and st_mtime_ns.
    """

    relative_path: str
    size_bytes: int
    modified_ns: int


@dataclass(frozen=True)
class SliceFingerprint:
    """What a built slice file was built from: the metric's label
    version and each selected partition's stamp, in partition order.

    Stored as JSON in the built file's schema metadata.
    """

    build_version: int
    label_version: int
    stamps: tuple[PartitionStamp, ...]

    @classmethod
    def of_partitions(
        cls,
        label_version: int,
        partitions: list[SeventeenLandsPartition],
        metrics_root: Path,
    ) -> SliceFingerprint:
        """The fingerprint of partitions as they are on disk now, under
        the current SLICE_BUILD_VERSION.

        Inputs: label_version (the metric's LABEL_VERSION, 0 for a row
            stream), partitions, metrics_root.
        Output: SliceFingerprint.
        Side effects: stats each partition file.
        Exceptions: FileNotFoundError if a partition file vanished.
        """
        stamps: list[PartitionStamp] = []
        for partition in partitions:
            path = partition.path(metrics_root)
            stat = path.stat()
            stamps.append(
                PartitionStamp(
                    path.relative_to(metrics_root).as_posix(),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            )
        return cls(SLICE_BUILD_VERSION, label_version, tuple(stamps))

    @classmethod
    def read_from(cls, path: Path) -> SliceFingerprint | None:
        """The fingerprint stored in a built slice file.

        Inputs: path. Output: the fingerprint, or None if path is
            missing or carries none.
        Side effects: opens path's footer (no row data).
        Exceptions: none (an unreadable file reads as None: rebuilt).
        """
        try:
            metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
            raw = json.loads(metadata[_FINGERPRINT_KEY])
            return cls(
                build_version=int(raw["build_version"]),
                label_version=int(raw["label_version"]),
                stamps=tuple(PartitionStamp(*stamp) for stamp in raw["stamps"]),
            )
        except (OSError, KeyError, TypeError, ValueError, pa.ArrowException):
            return None

    def schema_metadata(self) -> dict[bytes, bytes]:
        """This fingerprint as one schema-metadata entry.

        Inputs: none. Output: {b"seventeenlands_slice_inputs": JSON}.
        Side effects: none. Exceptions: none.
        """
        payload = {
            "build_version": self.build_version,
            "label_version": self.label_version,
            "stamps": [list(stamp) for stamp in self.stamps],
        }
        return {_FINGERPRINT_KEY: json.dumps(payload).encode("utf-8")}


class SeventeenLandsSliceFile:
    """Builds and caches slice files for one 17lands metric."""

    def __init__(
        self, metric: SlicedMetricClass, metrics_root: Path = METRICS_ROOT
    ) -> None:
        """
        Inputs:
            metric: the metric class (its FAMILY and OUTPUT_STEM name
                the partitions; its kind picks the build).
            metrics_root: where the partitions live and slice files go.
        Output: none (constructor).
        Side effects: none.
        Exceptions: TypeError if metric is not exactly one kind
            (is_count_table()).
        """
        self._metric = metric
        self._metrics_root = metrics_root
        self._count_metric = metric if is_count_table(metric) else None

    def path_for(self, data_slice: SeventeenLandsSlice) -> Path:
        """Where data_slice's built file lives (built or not).

        Inputs: data_slice.
        Output: <metrics_root>/seventeenlands/<FAMILY>/slices/
            <OUTPUT_STEM>.<data_slice.name>.parquet.
        Side effects: none. Exceptions: none.
        """
        file_name = f"{self._metric.OUTPUT_STEM}.{data_slice.name}.parquet"
        return _slices_directory(self._metric, self._metrics_root) / file_name

    def build(self, data_slice: SeventeenLandsSlice) -> Path:
        """The slice file for data_slice, built or rebuilt if stale.

        Inputs: data_slice.
        Output: path_for(data_slice).
        Side effects: if the file is missing or its fingerprint differs
            from the selected partitions', (re)writes it, creating
            parent directories. Otherwise none (no partition is read).
        Exceptions: ValueError if no partition matches data_slice, or
            the matching partitions disagree on MetricVersionMetadata
            (or one carries none); whatever pyarrow raises reading a
            corrupt partition.

        Example:
            >>> SeventeenLandsSliceFile(WinRateWhenInDeckMetric).build(SeventeenLandsSlice())
            PosixPath('data/metrics/seventeenlands/game_data/slices/win_rate_when_in_deck.all.parquet')
        """
        result = self.path_for(data_slice)

        # Select this slice's partitions
        partitions = [
            partition
            for partition in find_partitions(
                self._metric.FAMILY, self._metric.OUTPUT_STEM, self._metrics_root
            )
            if data_slice.includes(partition)
        ]
        if not partitions:
            raise ValueError(
                f"no {self._metric.OUTPUT_STEM} partitions match slice {data_slice.name}"
            )

        # Reuse a built file whose inputs haven't changed
        fingerprint = SliceFingerprint.of_partitions(
            self._label_version(), partitions, self._metrics_root
        )
        if SliceFingerprint.read_from(result) == fingerprint:
            return result

        # Build: every partition must come from one binder version
        version_metadata = self._shared_version_metadata(partitions)
        metadata = _built_file_metadata(version_metadata, fingerprint)
        if self._count_metric is not None:
            table = _count_table_output(
                self._count_metric, partitions, self._metrics_root
            )
            _write_atomically(
                result,
                lambda path: pq.write_table(
                    table.replace_schema_metadata(metadata), path
                ),
            )
        else:
            _write_atomically(
                result, lambda path: self._stream_rows_into(partitions, metadata, path)
            )
        return result

    def _label_version(self) -> int:
        """The metric's LABEL_VERSION, or 0 for a row stream.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        if self._count_metric is None:
            return 0
        return self._count_metric.LABEL_VERSION

    def _shared_version_metadata(
        self, partitions: list[SeventeenLandsPartition]
    ) -> MetricVersionMetadata:
        """The one MetricVersionMetadata every partition carries.

        Inputs: partitions (non-empty). Output: MetricVersionMetadata.
        Side effects: opens each partition's footer.
        Exceptions: ValueError naming the first partition that carries
            none or differs from the first's.
        """
        result: MetricVersionMetadata | None = None
        for partition in partitions:
            path = partition.path(self._metrics_root)
            metadata = read_version_metadata(path)
            if metadata is None:
                raise ValueError(f"{path} carries no version metadata")
            if result is not None and metadata != result:
                raise ValueError(f"{path} was built from {metadata}, not {result}")
            result = metadata
        if result is None:
            raise ValueError("no partitions to read version metadata from")
        return result

    def _stream_rows_into(
        self,
        partitions: list[SeventeenLandsPartition],
        metadata: dict[bytes, bytes],
        output_path: Path,
    ) -> None:
        """Concatenate a row stream's partitions into output_path, batch
        by batch, adding SET_COLUMN and FORMAT_COLUMN.

        Inputs: partitions (non-empty), metadata (the built file's
            schema metadata), output_path.
        Output: none.
        Side effects: streams every partition (never whole in memory);
            writes output_path.
        Exceptions: ValueError if two partitions' schemas differ
            (ignoring metadata).
        """
        source_schema = _bare_schema(partitions[0].path(self._metrics_root))
        output_schema = source_schema.append(pa.field(SET_COLUMN, pa.string())).append(
            pa.field(FORMAT_COLUMN, pa.string())
        )

        with pq.ParquetWriter(
            output_path, output_schema.with_metadata(metadata)
        ) as writer:
            # Each partition's batches, tagged with its set and format
            for partition in partitions:
                path = partition.path(self._metrics_root)
                if _bare_schema(path) != source_schema:
                    raise ValueError(f"{path}'s schema differs from {partitions[0]}")
                for batch in pq.ParquetFile(path).iter_batches():
                    writer.write_batch(_tagged_batch(batch, partition, output_schema))


def _count_table_output(
    metric: type[CountTableMetric],
    partitions: list[SeventeenLandsPartition],
    metrics_root: Path,
) -> pa.Table:
    """Sum partitions' counts per key and return the metric's
    output_from_counts() table.

    Inputs: metric, partitions (non-empty), metrics_root.
    Output: the finished table (no schema metadata yet).
    Side effects: reads every partition whole (count tables are one row
        per key).
    Exceptions: as finished_count_table, naming the partition whose
        columns are wrong.
    """
    paths = [partition.path(metrics_root) for partition in partitions]
    try:
        return finished_count_table(metric, [pq.read_table(path) for path in paths])
    except ValueError as error:
        raise ValueError(f"{metric.__name__} partitions {paths}: {error}") from error


def finished_count_table(
    metric: type[CountTableMetric], tables: Sequence[pa.Table]
) -> pa.Table:
    """Sum count-table partitions per key and label them: the table a
    dojo reads for the slice those partitions make up.

    Inputs: metric, tables (one or more partition tables, each exactly
        KEY_COLUMNS + COUNT_COLUMNS; schema metadata is ignored).
    Output: metric.output_from_counts() of the summed counts (and the
        summed baseline rows, for a HAS_BASELINE metric).
    Side effects: none.
    Exceptions: ValueError if tables is empty, a table's columns are not
        the metric's, a row has only some keys null, or baseline rows
        are missing (or present without HAS_BASELINE).

    Example:
        >>> finished_count_table(DrawnWinRateMetric, [pq.read_table(partition_path)])
    """
    expected = [*metric.KEY_COLUMNS, *metric.COUNT_COLUMNS]
    if not tables:
        raise ValueError(f"{metric.__name__}: no count tables to sum")
    for table in tables:
        if table.column_names != expected:
            raise ValueError(f"columns {table.column_names}, not {expected}")

    # Baseline rows aside; sum the rest per key
    combined = pa.concat_tables(
        [table.replace_schema_metadata(None) for table in tables],
        promote_options="permissive",
    )
    keyed, baseline = _split_baseline_rows(combined, metric.KEY_COLUMNS)
    return metric.output_from_counts(
        _sum_counts_per_key(keyed, metric), _sum_baseline(baseline, metric)
    )


def _built_file_metadata(
    version_metadata: MetricVersionMetadata, fingerprint: SliceFingerprint
) -> dict[bytes, bytes]:
    """A built slice file's schema metadata: the version keys plus the
    fingerprint.

    Inputs: version_metadata, fingerprint. Output: bytes-keyed dict.
    Side effects: none. Exceptions: none.
    """
    stamped = schema_with_version_metadata(pa.schema([]), version_metadata)
    return {**(stamped.metadata or {}), **fingerprint.schema_metadata()}


def _write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Run write against a temporary sibling of path, then move it into
    place, so a failed build never leaves a half-written slice file.

    Inputs: path, write (writes one file at the path it is given).
    Output: none.
    Side effects: creates path's parent directories; replaces path.
    Exceptions: whatever write raises (the temporary file is removed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bare_schema(path: Path) -> pa.Schema:
    """path's arrow schema without its metadata.

    Inputs: path. Output: pa.Schema.
    Side effects: opens path's footer.
    Exceptions: whatever pyarrow raises for a corrupt file.
    """
    return pq.ParquetFile(path).schema_arrow.remove_metadata()


def _tagged_batch(
    batch: pa.RecordBatch, partition: SeventeenLandsPartition, schema: pa.Schema
) -> pa.RecordBatch:
    """batch plus constant SET_COLUMN / FORMAT_COLUMN columns.

    Inputs: batch, partition (its set and format), schema (the output
        schema, without metadata).
    Output: pa.RecordBatch matching schema.
    Side effects: none. Exceptions: none.
    """
    rows = batch.num_rows
    columns = [
        *batch.columns,
        pa.array([partition.expansion.value] * rows, pa.string()),
        pa.array([partition.format.value] * rows, pa.string()),
    ]
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def _split_baseline_rows(
    table: pa.Table, key_columns: tuple[str, ...]
) -> tuple[pa.Table, pa.Table]:
    """table split into (keyed rows, null-key baseline rows).

    Inputs: table, key_columns. Output: (keyed, baseline).
    Side effects: none.
    Exceptions: ValueError if a row has some key columns null and
        others not.
    """
    nulls = [table.column(key).is_null() for key in key_columns]
    all_null = nulls[0]
    any_null = nulls[0]
    for column_nulls in nulls[1:]:
        all_null = pc.and_(all_null, column_nulls)
        any_null = pc.or_(any_null, column_nulls)
    if pc.any(pc.xor(all_null, any_null)).as_py():
        raise ValueError(f"a row has only some of {key_columns} null")
    return table.filter(pc.invert(all_null)), table.filter(all_null)


def _sum_counts_per_key(keyed: pa.Table, metric: type[CountTableMetric]) -> pa.Table:
    """keyed's COUNT_COLUMNS summed per KEY_COLUMNS value.

    Inputs: keyed (no baseline rows), metric.
    Output: KEY_COLUMNS + COUNT_COLUMNS, one row per key, named as in
        the partitions (not pyarrow's "<col>_sum"), sorted by key.
    Side effects: none. Exceptions: none.
    """
    keys = list(metric.KEY_COLUMNS)
    grouped = keyed.group_by(keys).aggregate(
        [(column, "sum") for column in metric.COUNT_COLUMNS]
    )
    result = grouped.select(
        [*keys, *(f"{column}_sum" for column in metric.COUNT_COLUMNS)]
    ).rename_columns([*keys, *metric.COUNT_COLUMNS])
    return result.sort_by([(key, "ascending") for key in keys])


def _sum_baseline(
    baseline: pa.Table, metric: type[CountTableMetric]
) -> pa.Table | None:
    """The slice's one-row baseline, or None if metric has none.

    Inputs: baseline (every partition's baseline rows), metric.
    Output: one row of COUNT_COLUMNS, or None unless HAS_BASELINE.
    Side effects: none.
    Exceptions: ValueError if HAS_BASELINE and baseline is empty, or
        not HAS_BASELINE and baseline is not.
    """
    if not metric.HAS_BASELINE:
        if baseline.num_rows:
            raise ValueError(f"{metric.__name__} has baseline rows but no baseline")
        return None
    if not baseline.num_rows:
        raise ValueError(f"{metric.__name__}'s partitions carry no baseline rows")
    counts = baseline.select(list(metric.COUNT_COLUMNS))
    sums = {
        column: [pc.sum(counts.column(column)).as_py()]
        for column in counts.column_names
    }
    return pa.table(sums, schema=counts.schema)


def _slices_directory(metric: SlicedMetricClass, metrics_root: Path) -> Path:
    """<metrics_root>/seventeenlands/<FAMILY>/slices.

    Inputs: metric, metrics_root. Output: Path.
    Side effects: none. Exceptions: none.
    """
    return metrics_root / SOURCE_DIRECTORY / metric.FAMILY.value / SLICES_DIRECTORY
