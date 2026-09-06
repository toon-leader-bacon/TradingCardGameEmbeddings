import contextlib
from pathlib import Path
import random
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.dojos.file_managers.utils.split_postfix import get_split_file_postfix
from src.dojos.file_managers.utils.TTVSplits import TTVSplits
from src.schema.splits import Split

MAX_INT = 2**31 - 1


class ParquetChunkReader:
    """Adapter: wraps a pyarrow `ParquetFile`'s row-group batch iteration
    so it yields pandas DataFrames, matching the shape callers already get
    from pandas' `TextFileReader` (FileManagerCSV's chunked reader) —
    `pandas.read_parquet` has no `chunksize=` equivalent, so this stands in
    for it.
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
    ) -> None:
        """
        Inputs:
            path_to_training_data: source .parquet file (e.g. a metric
                scanner's output, see AveragePickNumberMetric.finalize()).
            output_directory: directory the split files are written to.
            output_file_prefix: filename prefix for each split file.
            seed: RNG seed for shuffling; None means non-deterministic.
        Output: none (constructor).
        Side effects: opens path_to_training_data as a pyarrow ParquetFile
            to confirm it's actually readable, and caches its schema on
            self._schema for reuse by _stream_source_into_splits.
        Exceptions:
            FileNotFoundError if path_to_training_data doesn't exist.
            ValueError if it doesn't have a .parquet suffix, or is empty.
            Whatever pyarrow.parquet.ParquetFile raises for a corrupt or
            non-parquet file (parity with FileManagerCSV's __init__,
            which similarly reads nrows=1 to confirm the CSV parses).

        Note: unlike FileManagerCSV, there is no header_row concept —
        parquet always carries its own schema.
        """
        self.path_to_training_data = path_to_training_data
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.output_directory = output_directory
        self.output_file_prefix = output_file_prefix

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

    def make_splits(
        self,
        split_ratios: List[float] = [8, 1, 1],
        delete_old_splits: bool = True,
        shuffle: bool = True,
        batch_size: int = 32,
        load_row_group_batch_size: int = 10_000,
    ) -> List[ParquetChunkReader]:
        """Stream the source parquet file in row-group batches, split each
        batch by ratio, append each slice to the matching split's parquet
        file, then return one ParquetChunkReader per split.

        Inputs:
            split_ratios: unnormalized ratios, see TTVSplits.from_unnormalized.
            delete_old_splits: remove pre-existing split files first.
            shuffle: shuffle the rows within each streamed batch before
                splitting (same caveat as FileManagerCSV: this shuffles
                within a batch, not across the whole file).
            batch_size: rows per chunk in the returned readers.
            load_row_group_batch_size: rows read from the source file at
                a time while streaming.
        Output: one ParquetChunkReader per split, in split_ratios order.
        Side effects: creates/overwrites this instance's split files under
            output_directory.
        Exceptions: whatever pyarrow raises for a malformed source file.

        Composed of: _prepare_split_output_files, then
        _stream_source_into_splits (which itself calls
        _write_batch_to_splits per streamed batch).
        """
        splits: TTVSplits = TTVSplits.from_unnormalized(split_ratios)
        output_paths = self._prepare_split_output_files(
            splits.num_splits, delete_old_splits
        )
        self._stream_source_into_splits(
            output_paths, splits, shuffle, load_row_group_batch_size
        )
        return [ParquetChunkReader(path, batch_size) for path in output_paths]

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
        if split == Split.TRAIN:
            return self.shuffle_split_index(0, batch_size)
        elif split == Split.TEST:
            return self.shuffle_split_index(1, batch_size)
        elif split == Split.VALIDATION:
            return self.shuffle_split_index(2, batch_size)
        else:
            raise ValueError(f"Unsupported split: {split}")

    def shuffle_split_index(
        self, split_index: int, batch_size: int = 32
    ) -> ParquetChunkReader:
        """Load one split file fully into memory, shuffle its rows, write
        it back out, and return a fresh chunked reader over it.

        Inputs:
            split_index: which split file (0=train, 1=test, 2=validation,
                see get_split_file_postfix).
            batch_size: rows per chunk in the returned reader.
        Output: a ParquetChunkReader over the freshly-shuffled split file.
        Side effects: overwrites the target split file in place.
        Exceptions: FileNotFoundError if make_splits hasn't been run yet
            for this split_index.

        Reads with dtype_backend="pyarrow" (ArrowDtype columns) so the
        shuffle-and-rewrite round trip can't drift the file's on-disk
        schema away from self._schema over repeated epochs.
        """
        target_file = (
            self.output_directory
            / f"{self.output_file_prefix}_{get_split_file_postfix(split_index)}.parquet"
        )
        df = pd.read_parquet(target_file, dtype_backend="pyarrow")
        df = df.sample(frac=1, random_state=self.rng.randint(0, MAX_INT)).reset_index(
            drop=True
        )
        df.to_parquet(target_file, index=False)
        return ParquetChunkReader(target_file, batch_size)

    def _prepare_split_output_files(
        self, num_splits: int, delete_old_splits: bool = True
    ) -> List[Path]:
        """Compute each split's output path, deleting any pre-existing
        split files first when requested.

        Inputs:
            num_splits: how many split files to plan for.
            delete_old_splits: remove files matching this instance's
                output_file_prefix under output_directory first.
        Output: one Path per split, in split order (not yet written to —
            unlike FileManagerCSV, parquet has no header row to pre-write),
            named via the shared get_split_file_postfix.
        Side effects: creates output_directory if missing; deletes old
            split files when delete_old_splits is True.
        Exceptions: none expected.
        """
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if delete_old_splits:
            for path in self.output_directory.glob(
                f"{self.output_file_prefix}*.parquet"
            ):
                path.unlink()
        return [
            self.output_directory
            / f"{self.output_file_prefix}_{get_split_file_postfix(i)}.parquet"
            for i in range(num_splits)
        ]

    def _stream_source_into_splits(
        self,
        output_paths: List[Path],
        splits: TTVSplits,
        shuffle: bool,
        load_row_group_batch_size: int,
    ) -> None:
        """Stream self.path_to_training_data in row-group batches and
        write each batch's split-ratio slices into the matching split's
        parquet file.

        A ParquetWriter is a stateful resource — it only writes a valid
        parquet footer once closed, so a writer left open after an
        exception produces a corrupt, unreadable file. This method opens
        every split's writer through a single contextlib.ExitStack, so
        all of them are guaranteed to close together on the way out,
        success or failure — the parquet analog of CSV's simpler
        to_csv(mode="a") append, since parquet can't be appended to
        directly.

        Inputs:
            output_paths: this instance's split file paths, in split order.
            splits: precomputed split ratios/indices.
            shuffle: shuffle each streamed batch's rows before slicing.
            load_row_group_batch_size: rows read from the source file at
                a time while streaming.
        Output: none.
        Side effects: creates/writes every path in output_paths.
        Exceptions: whatever pyarrow raises for a malformed source file.

        Calls _write_batch_to_splits once per streamed batch.
        """
        source_file = pq.ParquetFile(self.path_to_training_data)
        with contextlib.ExitStack() as stack:
            writers = [
                stack.enter_context(pq.ParquetWriter(path, self._schema))
                for path in output_paths
            ]
            for record_batch in source_file.iter_batches(
                batch_size=load_row_group_batch_size
            ):
                batch = record_batch.to_pandas(types_mapper=pd.ArrowDtype)
                self._write_batch_to_splits(batch, writers, splits, shuffle)

    def _write_batch_to_splits(
        self,
        batch: pd.DataFrame,
        writers: List[pq.ParquetWriter],
        splits: TTVSplits,
        shuffle: bool,
    ) -> None:
        """Split one streamed batch by ratio and write each non-empty
        slice to its corresponding open writer.

        Inputs:
            batch: one row-group-sized chunk read from the source file.
            writers: this instance's open per-split writers, same order
                as splits.
            splits: precomputed split ratios/indices for this batch.
            shuffle: shuffle batch's rows before slicing.
        Output: none.
        Side effects: writes to each writer in `writers`.
        Exceptions: pyarrow.lib.ArrowInvalid if a slice's dtypes can't be
            cast to self._schema — shouldn't occur in normal use, since
            `batch` arrives with ArrowDtype columns (see
            _stream_source_into_splits) that already match the source
            schema exactly, nulls included.
        """
        if shuffle:
            batch = batch.sample(
                frac=1,
                random_state=self.rng.randint(0, MAX_INT),
            ).reset_index(drop=True)

        for i, (start_index, end_index) in enumerate(
            splits.get_split_indices(len(batch))
        ):
            if start_index == end_index:
                continue
            slice_df = batch.iloc[start_index:end_index]
            writers[i].write_table(
                pa.Table.from_pandas(
                    slice_df,
                    schema=self._schema,
                    preserve_index=False,
                )
            )
