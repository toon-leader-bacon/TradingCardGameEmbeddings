import contextlib
import hashlib
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.dojos.file_managers.bucket_shuffle import ShufflePlan, scatter_into_buckets
from src.schema.splits import Split
from src.schema.ttv_splits import TTVSplits

MAX_INT = 2**31 - 1

# Split file order written by make_splits (see _split_file_postfix).
_SPLIT_INDEX = {Split.TRAIN: 0, Split.TEST: 1, Split.VALIDATION: 2}

# Schema metadata every split file make_splits writes carries: its rows are
# in uniformly random order. splits_exist() treats a file without it (an
# older, order-preserving split) as missing, so it is rebuilt.
_SPLIT_ORDER_KEY = b"nocab_split_order"
_UNIFORM_SHUFFLE = b"uniform_shuffle"

# Uncompressed bytes per shuffle bucket: one bucket is in memory at a time,
# a few copies of it while it is shuffled and written
_DEFAULT_BUCKET_BYTES = 256 * 2**20

# Rows per row group in a split file. A reader decodes one row group at a
# time, so this bounds each open reader's memory (one per dojo in a run)
_SPLIT_ROW_GROUP_ROWS = 8_000

# make_splits' temporaries (bucket directories, partial split files) live in
# this subdirectory of the output directory, out of the way of the splits
# and on the same disk, so a partial file is renamed into place, not copied
_SHUFFLE_TEMP_DIRECTORY = ".shuffle_tmp"
_PARTIAL = ".partial"
_BUCKETS = "_buckets_"

# The single list of named split-file postfixes, by split index; any later
# index is "split_<index>". _split_file_postfix and _SPLIT_FILE_NAME both
# derive from it
_NAMED_POSTFIXES = ("train", "test", "validation")

# A split file's name, given its escaped prefix (see _split_file_postfix)
_SPLIT_FILE_NAME = (
    r"{}_(?:" + "|".join(map(re.escape, _NAMED_POSTFIXES)) + r"|split_\d+)\.parquet"
)


def _split_file_postfix(split_index: int) -> str:
    """Filename postfix for one split file.

    Inputs: split_index (int), position in make_splits' split_ratios.
    Output: str, _NAMED_POSTFIXES[split_index] ("train"/"test"/
        "validation" for 0/1/2), else "split_{index}".
    Side effects: none. Exceptions: none.
    """
    if split_index < len(_NAMED_POSTFIXES):
        return _NAMED_POSTFIXES[split_index]
    return f"split_{split_index}"


def _is_split_file_name(name: str, prefix: str) -> bool:
    """Whether name is exactly a split file of prefix,
    "<prefix>_<postfix>.parquet" with a _split_file_postfix postfix. Exact,
    so prefix "a.win_rate" never claims "a.win_rate_at_act2_train.parquet"
    (another dojo's split).

    Inputs: name (a file name), prefix. Output: bool.
    Side effects: none. Exceptions: none.

    Example:
        >>> _is_split_file_name("x_train.parquet", "x")
        True
        >>> _is_split_file_name("x_y_train.parquet", "x")
        False
    """
    return re.fullmatch(_SPLIT_FILE_NAME.format(re.escape(prefix)), name) is not None


def _read_whole(path: Path) -> pd.DataFrame:
    """A whole parquet file (a bucket, or a source small enough for one
    bucket) as a DataFrame with ArrowDtype columns, so nullable ints keep
    their nulls (see ParquetChunkReader.__next__).

    Inputs: path. Output: pd.DataFrame. Side effects: reads the file.
    Exceptions: whatever pyarrow raises for a missing or corrupt file.
    """
    return pq.read_table(path).to_pandas(types_mapper=pd.ArrowDtype)


def _partial_path_of(split_path: Path) -> Path:
    """Where make_splits writes split_path before it is complete:
    "<name>.partial" in the output directory's _SHUFFLE_TEMP_DIRECTORY,
    where neither splits_exist() nor the split-file cleanup looks.

    Inputs: split_path. Output: Path. Side effects: none. Exceptions: none.

    Example:
        >>> _partial_path_of(Path("out/x_train.parquet"))
        PosixPath('out/.shuffle_tmp/x_train.parquet.partial')
    """
    return split_path.parent / _SHUFFLE_TEMP_DIRECTORY / f"{split_path.name}{_PARTIAL}"


@contextlib.contextmanager
def _deleted_on_failure(paths: List[Path]) -> Iterator[None]:
    """Context manager: if the block raises any BaseException (so Ctrl-C's
    KeyboardInterrupt too, not only Exception), delete whichever of paths
    exist, then re-raise; on success, keep them.

    Inputs: paths. Output: a context manager yielding None.
    Side effects: on failure, deletes files.
    Exceptions: re-raises the block's exception.
    """
    try:
        yield
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise


def _remove_if_empty(directory: Path) -> None:
    """Delete directory if it is empty (make_splits' temporary directory
    after a clean run); leave it otherwise.

    Inputs: directory (exists). Output: none.
    Side effects: may delete the directory. Exceptions: none.
    """
    if not any(directory.iterdir()):
        directory.rmdir()


def _has_uniform_order(split_path: Path) -> bool:
    """Whether split_path exists and its schema metadata carries
    _SPLIT_ORDER_KEY: _UNIFORM_SHUFFLE (written by the current make_splits).

    Inputs: split_path. Output: bool (False for a missing file).
    Side effects: reads the file's footer.
    Exceptions: whatever pyarrow raises for a corrupt file.
    """
    if not split_path.exists():
        return False
    metadata = pq.read_schema(split_path).metadata or {}
    return metadata.get(_SPLIT_ORDER_KEY) == _UNIFORM_SHUFFLE


def _group_split_positions(
    group_values: pd.Series, salt: str, splits: TTVSplits
) -> np.ndarray:
    """Which split each row's group lands in, the same for every row of
    one group.

    Each distinct group value is hashed with salt to a fraction in
    [0, 1), and the fraction picks the split by cumulative ratio. The
    split depends only on (salt, value), never on row order or batch,
    so a group spread across streamed batches still lands in one split.

    Inputs:
        group_values: one batch's group column (compared by str()).
        salt: seeds the hash; the same salt reproduces the assignment.
        splits: the split ratios.
    Output: np.ndarray[int], one split position per row (0 = first
        ratio).
    Side effects: none. Exceptions: none.

    Example:
        >>> positions = _group_split_positions(
        ...     pd.Series(["a", "a", "b"]), "7", TTVSplits.from_unnormalized([8, 1, 1])
        ... )
        >>> positions[0] == positions[1]
        True
    """
    distinct_values = pd.unique(group_values)
    fractions = np.array(
        [
            int.from_bytes(
                hashlib.blake2b(f"{salt}:{value}".encode(), digest_size=8).digest(),
                "big",
            )
            / 2**64
            for value in distinct_values
        ]
    )
    cumulative = np.cumsum(splits.get_percentages())
    positions = np.minimum(
        np.searchsorted(cumulative, fractions, side="right"), splits.num_splits - 1
    )
    position_of_value = dict(zip(distinct_values, positions))
    return np.array([position_of_value[value] for value in group_values], dtype=int)


class ParquetChunkReader:
    """Adapter: wraps a pyarrow `ParquetFile`'s row-group batch iteration
    so it yields pandas DataFrames, like pandas' chunked `TextFileReader`
    does for CSV — `pandas.read_parquet` has no `chunksize=` equivalent, so
    this stands in for it.
    """

    def __init__(self, path: Path, batch_size: int) -> None:
        """
        Inputs:
            path: parquet file to stream.
            batch_size: rows per yielded DataFrame.
        Output: none (constructor).
        Side effects: opens the file for reading (no data loaded yet).
        Exceptions: whatever pyarrow.parquet.ParquetFile raises for a
            missing/corrupt file.
        """
        self._parquet_file = pq.ParquetFile(path)
        self._batches = self._parquet_file.iter_batches(batch_size=batch_size)

    def __iter__(self) -> "ParquetChunkReader":
        """Output: self — this reader is its own iterator."""
        return self

    def __next__(self) -> pd.DataFrame:
        """
        Output: the next batch_size-row chunk as a DataFrame. Columns use
            pandas' ArrowDtype (types_mapper=pd.ArrowDtype) rather than
            plain numpy dtypes, so a nullable integer column round-trips
            through pandas without silently upcasting to float64 (numpy
            has no nullable int, so a null-containing batch would
            otherwise turn e.g. an int64 column into float64+NaN).
        Exceptions: StopIteration once the file is exhausted.
        """
        return next(self._batches).to_pandas(types_mapper=pd.ArrowDtype)


class FileManagerParquet:

    _schema: pa.Schema

    def __init__(
        self,
        path_to_training_data: Path,
        output_directory: Path,
        output_file_prefix: str = "split_",
        seed: int | None = None,
        split_group_column: str | None = None,
    ) -> None:
        """
        Inputs:
            path_to_training_data: source .parquet file (e.g. a metric
                parquet output).
            output_directory: directory the split files are written to.
            output_file_prefix: filename prefix for each split file.
            seed: RNG seed for shuffling and the group-split hash; None
                means non-deterministic.
            split_group_column: None splits row by row. A column name
                splits by that column's value instead: every row
                sharing a value (e.g. one deck_uuid) lands in the same
                split, so rows about one deck or kingdom never sit in
                both TRAIN and TEST.
        Output: none (constructor).
        Side effects: opens path_to_training_data as a pyarrow ParquetFile
            to confirm it's actually readable, and caches its schema on
            self._schema (the bucket files' schema, and the split files'
            before _split_schema adds its stamp).
        Exceptions:
            FileNotFoundError if path_to_training_data doesn't exist.
            ValueError if it doesn't have a .parquet suffix, is empty, or
            split_group_column is not one of its columns.
            Whatever pyarrow.parquet.ParquetFile raises for a corrupt or
            non-parquet file.
        """
        self.path_to_training_data = path_to_training_data
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.output_directory = output_directory
        self.output_file_prefix = output_file_prefix
        self._split_group_column = split_group_column
        # From seed directly, not self.rng: a group's split depends only on
        # (seed, value), never on how many shuffle draws came before.
        self._group_salt = str(seed) if seed is not None else str(random.random())

        if not self.path_to_training_data.exists():
            raise FileNotFoundError(
                f"Training data file not found: {self.path_to_training_data}"
            )

        if not self.path_to_training_data.suffix == ".parquet":
            raise ValueError(
                f"Training data file is not a parquet file: {self.path_to_training_data}"
            )

        if self.path_to_training_data.stat().st_size == 0:
            raise ValueError(
                f"Training data file is empty: {self.path_to_training_data}"
            )

        self._schema = pq.ParquetFile(self.path_to_training_data).schema_arrow
        if (
            split_group_column is not None
            and split_group_column not in self._schema.names
        ):
            raise ValueError(
                f"split_group_column {split_group_column!r} is not a column of "
                f"{self.path_to_training_data} (columns: {self._schema.names})"
            )

    @property
    def schema(self) -> pa.Schema:
        """The source file's cached schema, metadata included.

        Exposes what __init__ already computed (self._schema) so a
        caller (e.g. GenericDojo's version-metadata check) can read it
        without reaching into a private attribute or re-opening the
        file. Every split file make_splits() writes carries this same
        schema plus one metadata key, _SPLIT_ORDER_KEY (_split_schema).
        """
        return self._schema

    @property
    def _split_schema(self) -> pa.Schema:
        """self._schema with _SPLIT_ORDER_KEY: _UNIFORM_SHUFFLE added to its
        metadata; what every split file is written with.

        Inputs: none. Output: pa.Schema. Side effects: none.
        Exceptions: none.
        """
        metadata = dict(self._schema.metadata or {})
        metadata[_SPLIT_ORDER_KEY] = _UNIFORM_SHUFFLE
        return self._schema.with_metadata(metadata)

    @property
    def _shuffle_temp_directory(self) -> Path:
        """output_directory / _SHUFFLE_TEMP_DIRECTORY, where make_splits'
        temporaries live. Inputs: none. Output: Path. Side effects: none.
        Exceptions: none."""
        return self.output_directory / _SHUFFLE_TEMP_DIRECTORY

    def make_splits(
        self,
        split_ratios: List[float] = [8, 1, 1],
        delete_old_splits: bool = True,
        batch_size: int = 32,
        load_row_group_batch_size: int = 10_000,
        bucket_bytes: int = _DEFAULT_BUCKET_BYTES,
    ) -> List[ParquetChunkReader]:
        """Write the source's rows, in uniformly random order, into one
        parquet file per split, then return a ParquetChunkReader per split.

        The shuffle is over the whole file in bounded memory: a source
        bigger than bucket_bytes in memory is first scattered into
        temporary bucket files at random (bucket_shuffle.py), then each
        bucket is shuffled whole and appended to the splits; a smaller one
        is shuffled whole directly. Rows split by ratio, or by
        split_group_column's hash (_split_slices).

        A stamped split file is always a complete one: the splits are
        written as partial files in output_directory/.shuffle_tmp and
        renamed into place only after every one of them finished. A
        failure (or Ctrl-C) while writing deletes the partial files and
        leaves the old splits untouched. A failure or kill while swapping
        (deleting the old splits, then renaming) can leave a final name
        missing and partial files behind; splits_exist() is then False,
        and the next build sweeps the partials and starts over.

        Inputs:
            split_ratios: unnormalized ratios, see TTVSplits.from_unnormalized.
            delete_old_splits: also remove this prefix's other split files
                ("<prefix>_split_3.parquet" from an older 4-way split); the
                splits being replaced always go.
            batch_size: rows per chunk in the returned readers.
            load_row_group_batch_size: the fewest source rows read per
                scatter step (>= 1; ShufflePlan raises it for many
                buckets).
            bucket_bytes: Arrow (in-memory) bytes per bucket (> 0),
                estimated from a sample of rows; bounds memory.
        Output: one ParquetChunkReader per split, in split_ratios order.
        Side effects: replaces this instance's split files under
            output_directory, each stamped _SPLIT_ORDER_KEY; creates and
            removes partial files and a bucket directory in
            output_directory/.shuffle_tmp (the directory too, once empty),
            and removes this prefix's temporaries a killed earlier run left.
        Exceptions: ValueError if bucket_bytes <= 0 or
            load_row_group_batch_size < 1; whatever pyarrow raises for a
            malformed source file; OSError if an old split cannot be
            deleted or a partial file renamed (e.g. an old split still
            open in a reader, on Windows).

        Example:
            >>> readers = FileManagerParquet(source, out, seed=0).make_splits([8, 1, 1])
            >>> len(readers)
            3
        """
        result: List[ParquetChunkReader]

        # Plan the shuffle from a sample of the source (validates the sizes)
        splits: TTVSplits = TTVSplits.from_unnormalized(split_ratios)
        plan = ShufflePlan.for_source(
            pq.ParquetFile(self.path_to_training_data),
            bucket_bytes,
            load_row_group_batch_size,
        )

        # Write every split under its partial name (deleted on failure)
        self._shuffle_temp_directory.mkdir(parents=True, exist_ok=True)
        self._remove_stale_temporaries()
        final_paths = [self._split_path(index) for index in range(splits.num_splits)]
        partial_paths = [_partial_path_of(path) for path in final_paths]
        self._write_shuffled_splits(partial_paths, splits, plan)

        # Only now replace the old splits: the stamp implies "complete"
        self._delete_old_splits(final_paths, every_prefixed_file=delete_old_splits)
        for partial_path, final_path in zip(partial_paths, final_paths):
            partial_path.replace(final_path)
        _remove_if_empty(self._shuffle_temp_directory)
        result = [ParquetChunkReader(path, batch_size) for path in final_paths]
        return result

    def splits_exist(self) -> bool:
        """Whether this instance's train/test/validation split files are
        on disk under output_directory/output_file_prefix, written by the
        current make_splits (stamped _SPLIT_ORDER_KEY: _UNIFORM_SHUFFLE).

        Output: True only if all three split files exist and are stamped.
            False if any is missing (including "make_splits has never run
            for this output_directory/output_file_prefix") or unstamped (an
            older split that kept the source's order, rebuilt by the
            caller).
        Side effects: reads each split file's footer (schema metadata).
        Exceptions: whatever pyarrow raises for a corrupt split file.
        """
        return all(
            _has_uniform_order(self._split_path(index))
            for index in _SPLIT_INDEX.values()
        )

    def reader_for(self, split: Split, batch_size: int = 32) -> ParquetChunkReader:
        """Open a fresh chunked reader over one already-written split file.

        Inputs:
            split: which split to read.
            batch_size: rows per chunk in the returned reader.
        Output: a ParquetChunkReader positioned at the split's first row.
        Side effects: opens the split file for reading.
        Exceptions: FileNotFoundError if make_splits hasn't been run yet.
        """
        return ParquetChunkReader(self._split_path(_SPLIT_INDEX[split]), batch_size)

    def row_count(self, split: Split) -> int:
        """Number of rows in one already-written split file.

        Inputs: split (Split). Output: int, from parquet metadata (no data
        read). Side effects: none. Exceptions: FileNotFoundError if
        make_splits hasn't been run yet.
        """
        return pq.ParquetFile(self._split_path(_SPLIT_INDEX[split])).metadata.num_rows

    def shuffle_split(self, split: Split, batch_size: int = 32) -> ParquetChunkReader:
        """Convenience wrapper around shuffle_split_index taking the real
        Split enum (src.schema.splits.Split) instead of a raw index — the
        interface the trainer uses to re-shuffle a split's data (typically
        the training split, between epochs).

        Inputs:
            split: which split to reshuffle.
            batch_size: rows per chunk in the returned reader.
        Output: a ParquetChunkReader over the freshly-shuffled split file.
        Side effects: see shuffle_split_index.
        Exceptions: ValueError for an unrecognized split.
        """
        if split not in _SPLIT_INDEX:
            raise ValueError(f"Unsupported split: {split}")
        return self.shuffle_split_index(_SPLIT_INDEX[split], batch_size)

    def split_path(self, split: Split) -> Path:
        """Where one split's parquet file lives (it may not exist yet).

        Inputs: split. Output: Path. Side effects: none. Exceptions: none.

        Example:
            >>> manager.split_path(Split.TRAIN)
            PosixPath('data/splits/rarity_tier_train.parquet')
        """
        return self._split_path(_SPLIT_INDEX[split])

    def _split_path(self, split_index: int) -> Path:
        return (
            self.output_directory
            / f"{self.output_file_prefix}_{_split_file_postfix(split_index)}.parquet"
        )

    def shuffle_split_index(
        self, split_index: int, batch_size: int = 32
    ) -> ParquetChunkReader:
        """Load one split file fully into memory, shuffle its rows, write
        it back out, and return a fresh chunked reader over it.

        Inputs:
            split_index: which split file (0=train, 1=test, 2=validation,
                see _split_file_postfix).
            batch_size: rows per chunk in the returned reader.
        Output: a ParquetChunkReader over the freshly-shuffled split file.
        Side effects: overwrites the target split file in place.
        Exceptions: FileNotFoundError if make_splits hasn't been run yet
            for this split_index.

        Reads with dtype_backend="pyarrow" (ArrowDtype columns) so the
        shuffle-and-rewrite round trip can't drift the file's on-disk
        schema away from self._schema over repeated epochs. Note: the
        rewrite drops make_splits' _SPLIT_ORDER_KEY stamp (and its row-group
        size), so splits_exist() then reports the splits missing and the
        next dojo build re-splits. Only tests call it today.
        """
        target_file = self._split_path(split_index)
        df = pd.read_parquet(target_file, dtype_backend="pyarrow")
        df = df.sample(frac=1, random_state=self.rng.randint(0, MAX_INT)).reset_index(
            drop=True
        )
        df.to_parquet(target_file, index=False)
        return ParquetChunkReader(target_file, batch_size)

    def _delete_old_splits(
        self, final_paths: List[Path], every_prefixed_file: bool
    ) -> None:
        """Remove the split files about to be replaced plus all three
        standard train/test/validation paths (so a one-split run never
        leaves an older stamped test file beside its new train file), and
        with every_prefixed_file, every other split file of this prefix
        (_is_split_file_name: exact names, so another dojo whose name
        starts with this prefix keeps its splits).

        Private helper - single caller is make_splits(), after the new
        splits are complete.
        Inputs: final_paths (this run's split paths), every_prefixed_file.
        Output: none. Side effects: deletes files. Exceptions: OSError if
            a file cannot be deleted.
        """
        doomed = set(final_paths)
        doomed.update(self._split_path(index) for index in _SPLIT_INDEX.values())
        if every_prefixed_file:
            doomed.update(
                path
                for path in self.output_directory.iterdir()
                if _is_split_file_name(path.name, self.output_file_prefix)
            )
        for path in doomed:
            path.unlink(missing_ok=True)

    def _remove_stale_temporaries(self) -> None:
        """Delete what a killed earlier make_splits left in
        output_directory/.shuffle_tmp (an exception already removes its
        own): this prefix's "<prefix>_buckets_*" directories (many GB for
        the 17lands sources) and partial split files.

        Private helper - single caller is make_splits().
        Inputs: none. Output: none. Side effects: deletes directories and
            files.
        Exceptions: OSError if one cannot be deleted.
        """
        prefix = self.output_file_prefix
        for entry in self._shuffle_temp_directory.iterdir():
            if entry.is_dir() and entry.name.startswith(f"{prefix}{_BUCKETS}"):
                shutil.rmtree(entry)
            elif entry.name.endswith(_PARTIAL) and _is_split_file_name(
                entry.name.removesuffix(_PARTIAL), prefix
            ):
                entry.unlink()

    def _write_shuffled_splits(
        self, output_paths: List[Path], splits: TTVSplits, plan: ShufflePlan
    ) -> None:
        """Pass 2 of the shuffle: every bucket (or the whole source, for a
        one-bucket plan) loaded, shuffled whole and appended to the splits.

        A ParquetWriter only writes a valid footer once closed, so every
        split writer, and the temporary bucket directory, is opened through
        one contextlib.ExitStack: all close (and the buckets are deleted)
        on the way out, success or failure. On failure the files at
        output_paths are deleted too (_deleted_on_failure): a closed
        writer's file is valid and stamped, but incomplete.

        Private helper - single caller is make_splits().
        Inputs: output_paths (the splits' partial paths, in split order),
            splits, plan.
        Output: none.
        Side effects: writes every path in output_paths (deleting them on
            failure); for a plan of more than one bucket, creates bucket
            files in a temporary directory under output_directory and
            removes it.
        Exceptions: whatever pyarrow raises for a malformed source file.
        """
        with _deleted_on_failure(output_paths), contextlib.ExitStack() as stack:
            writers = [
                stack.enter_context(pq.ParquetWriter(path, self._split_schema))
                for path in output_paths
            ]

            # A source that fits in one bucket is shuffled whole, no buckets
            if plan.bucket_count == 1:
                source = _read_whole(self.path_to_training_data)
                self._write_bucket_to_splits(source, writers, splits)
                return

            # Pass 1: scatter rows into buckets at random (same disk as the
            # splits; the system temp drive may be too small)
            bucket_directory = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(
                        dir=self._shuffle_temp_directory,
                        prefix=f"{self.output_file_prefix}{_BUCKETS}",
                    )
                )
            )
            bucket_paths = scatter_into_buckets(
                self.path_to_training_data,
                plan,
                self._schema,
                bucket_directory,
                seed=self.rng.randint(0, MAX_INT),
            )

            # Pass 2: each bucket shuffled whole, then split
            for bucket_path in bucket_paths:
                bucket = _read_whole(bucket_path)
                self._write_bucket_to_splits(bucket, writers, splits)

    def _write_bucket_to_splits(
        self,
        bucket: pd.DataFrame,
        writers: List[pq.ParquetWriter],
        splits: TTVSplits,
    ) -> None:
        """Shuffle one whole bucket (or source), cut it into its per-split
        rows (_split_slices: by ratio, or by group hash), and write each
        non-empty slice to its open writer in row groups of
        _SPLIT_ROW_GROUP_ROWS.

        Inputs:
            bucket: one bucket's rows (or the whole source's), ArrowDtype
                columns.
            writers: this instance's open per-split writers, same order
                as splits.
            splits: precomputed split ratios/indices.
        Output: none.
        Side effects: writes to each writer in `writers`; advances self.rng.
        Exceptions: pyarrow.lib.ArrowInvalid if a slice's dtypes can't be
            cast to self._split_schema — shouldn't occur in normal use,
            since `bucket` arrives with ArrowDtype columns (_read_whole)
            that already match the source schema exactly, nulls included.
        """
        shuffled = bucket.sample(
            frac=1, random_state=self.rng.randint(0, MAX_INT)
        ).reset_index(drop=True)
        for writer, slice_df in zip(writers, self._split_slices(shuffled, splits)):
            if slice_df.empty:
                continue
            writer.write_table(
                pa.Table.from_pandas(
                    slice_df, schema=self._split_schema, preserve_index=False
                ),
                row_group_size=_SPLIT_ROW_GROUP_ROWS,
            )

    def _split_slices(
        self, batch: pd.DataFrame, splits: TTVSplits
    ) -> List[pd.DataFrame]:
        """One bucket cut into its per-split rows.

        Inputs: batch (a whole bucket, already shuffled), splits.
        Output: one DataFrame per split, in split order (possibly
            empty). Row by row: contiguous slices by ratio. With a
            split group column: each row goes where its group hashes
            (_group_split_positions).
        Side effects: none. Exceptions: none.
        """
        if self._split_group_column is None:
            return [
                batch.iloc[start_index:end_index]
                for start_index, end_index in splits.get_split_indices(len(batch))
            ]
        positions = _group_split_positions(
            batch[self._split_group_column], self._group_salt, splits
        )
        return [batch[positions == i] for i in range(splits.num_splits)]
