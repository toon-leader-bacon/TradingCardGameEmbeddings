"""Pass 1 of FileManagerParquet.make_splits' out-of-core shuffle: scatter a
source parquet file's rows into temporary bucket files at random.

make_splits shuffles a source too big to hold in memory in two passes:

1. scatter_into_buckets streams the source and sends every row to one of
   ShufflePlan.bucket_count bucket files, chosen uniformly at random;
2. make_splits then loads each bucket whole, shuffles it, and appends it
   to the split files.

Random bucket assignment followed by a full shuffle inside each bucket
gives a uniformly random order of the whole file, while only one bucket
(about ShufflePlan's byte target) is ever in memory.
"""

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Most bucket files make_splits opens at once (one ParquetWriter each); a
# bigger source gets bigger buckets instead of more of them
_MAX_BUCKETS = 512

# Average rows per bucket per scatter write: fewer would leave each bucket
# file thousands of tiny row groups
_ROWS_PER_BUCKET_WRITE = 1_000

# Rows decoded to estimate a source's in-memory size. Parquet's own
# "uncompressed" sizes are after dictionary encoding, which can be 30x
# smaller than the Arrow table (17lands card-id lists: 13 vs 413 bytes/row)
_SIZE_SAMPLE_ROWS = 10_000


@dataclass(frozen=True)
class ShufflePlan:
    """How make_splits shuffles one source file in bounded memory.

    bucket_count: bucket files to scatter into (>= 1); 1 means the source
        fits in one bucket and is shuffled whole, with no temporary files.
    scatter_batch_rows: rows read from the source per scatter step (>= 1).
    """

    bucket_count: int
    scatter_batch_rows: int

    def __post_init__(self) -> None:
        """Side effects: none. Exceptions: ValueError if either field < 1."""
        if self.bucket_count < 1 or self.scatter_batch_rows < 1:
            raise ValueError(
                f"bucket_count and scatter_batch_rows must be >= 1, got "
                f"{self.bucket_count} and {self.scatter_batch_rows}"
            )

    @classmethod
    def for_source(
        cls, source_file: pq.ParquetFile, bucket_bytes: int, min_scatter_rows: int
    ) -> "ShufflePlan":
        """The plan for a source: enough buckets that each holds about
        bucket_bytes in memory (capped at _MAX_BUCKETS), and scatter
        batches large enough to average _ROWS_PER_BUCKET_WRITE rows per
        bucket per write.

        Inputs: source_file (the source, opened), bucket_bytes (> 0, of
            Arrow memory per bucket), min_scatter_rows (>= 1, the scatter
            batch floor).
        Output: ShufflePlan.
        Side effects: reads the source's first _SIZE_SAMPLE_ROWS rows.
        Exceptions: ValueError if bucket_bytes <= 0 or min_scatter_rows < 1.

        Example:
            >>> ShufflePlan.for_source(pq.ParquetFile(path), 256 * 2**20, 10_000)
            ShufflePlan(bucket_count=1, scatter_batch_rows=10000)
        """
        # Validate inputs
        if bucket_bytes <= 0 or min_scatter_rows < 1:
            raise ValueError(
                f"bucket_bytes must be > 0 and min_scatter_rows >= 1, got "
                f"{bucket_bytes} and {min_scatter_rows}"
            )

        # Enough buckets for the in-memory size, within the writer cap
        total_bytes = _estimated_memory_bytes(source_file)
        bucket_count = min(_MAX_BUCKETS, max(1, math.ceil(total_bytes / bucket_bytes)))

        # Batches big enough that each bucket write is not tiny
        scatter_batch_rows = max(
            min_scatter_rows, bucket_count * _ROWS_PER_BUCKET_WRITE
        )
        return cls(bucket_count, scatter_batch_rows)


def scatter_into_buckets(
    source: Path,
    plan: ShufflePlan,
    schema: pa.Schema,
    bucket_directory: Path,
    seed: int,
) -> list[Path]:
    """Pass 1: write every row of source to one of plan.bucket_count bucket
    files, each row's bucket drawn uniformly at random.

    Inputs: source (a parquet file), plan (its ShufflePlan), schema (written
        to every bucket; the source's own), bucket_directory (an existing,
        empty directory the caller deletes afterwards), seed (the bucket
        draws; the same seed gives the same buckets).
    Output: list[Path], one per bucket, in bucket order. Every bucket file
        exists and is valid, possibly with no rows.
    Side effects: writes plan.bucket_count files under bucket_directory.
    Exceptions: whatever pyarrow raises for a malformed source.

    Example:
        >>> paths = scatter_into_buckets(source, ShufflePlan(4, 10_000), schema, tmp, seed=0)
        >>> len(paths)
        4
    """
    result = [
        bucket_directory / f"bucket_{index:05d}.parquet"
        for index in range(plan.bucket_count)
    ]
    rng = np.random.default_rng(seed)
    source_file = pq.ParquetFile(source)

    # One writer per bucket, all closed together (a ParquetWriter left open
    # never writes its footer)
    with contextlib.ExitStack() as stack:
        writers = [
            stack.enter_context(pq.ParquetWriter(path, schema)) for path in result
        ]
        # Send each row of each streamed batch to a uniformly random bucket
        for record_batch in source_file.iter_batches(
            batch_size=plan.scatter_batch_rows
        ):
            bucket_of_row = rng.integers(plan.bucket_count, size=record_batch.num_rows)
            # The writers compare schemas without metadata, so the batch's
            # own (metadata-free) schema is accepted
            table = pa.Table.from_batches([record_batch])
            for bucket_index, rows in _rows_by_bucket(table, bucket_of_row):
                writers[bucket_index].write_table(rows)
    return result


def _estimated_memory_bytes(source_file: pq.ParquetFile) -> int:
    """The source's estimated size as Arrow tables: the bytes per row of
    its first _SIZE_SAMPLE_ROWS rows, times its row count.

    Private helper - single caller is ShufflePlan.for_source().
    Inputs: source_file. Output: int >= 0 (0 for a source with no rows).
    Side effects: reads the sample rows.
    Exceptions: whatever pyarrow raises for a malformed source.
    """
    total_rows = source_file.metadata.num_rows
    sample = next(source_file.iter_batches(batch_size=_SIZE_SAMPLE_ROWS), None)
    if sample is None or sample.num_rows == 0:
        return 0
    return math.ceil(sample.nbytes / sample.num_rows * total_rows)


def _rows_by_bucket(
    table: pa.Table, bucket_of_row: np.ndarray
) -> Iterator[tuple[int, pa.Table]]:
    """table's rows grouped by bucket: one (bucket index, rows) pair per
    bucket that received at least one row, rows in table order.

    Private helper - single caller is scatter_into_buckets(). Sorts the
    row indices by bucket (stable) and slices the taken table, rather than
    filtering once per bucket.
    Inputs: table, bucket_of_row (int array, one bucket per row).
    Output: iterator of (int, pa.Table). Side effects: none.
    Exceptions: none.
    """
    # Rows of one bucket made contiguous, each bucket keeping table order
    order = np.argsort(bucket_of_row, kind="stable")
    grouped = table.take(pa.array(order))
    counts = np.bincount(bucket_of_row)

    # Slice out each bucket that got rows
    start = 0
    for bucket_index, count in enumerate(counts):
        if count:
            yield bucket_index, grouped.slice(start, int(count))
            start += int(count)
