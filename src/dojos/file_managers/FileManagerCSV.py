from pathlib import Path
import random
from typing import List

import pandas as pd
from pandas.io.parsers.readers import TextFileReader

from src.dojos.dojo import Split
from src.dojos.file_managers.utils.TTVSplits import TTVSplits

MAX_INT = 2**31 - 1


class FileManagerCSV:

    def __init__(self, path_to_training_data: Path,
                 output_directory: Path,
                 output_file_prefix: str = "split_",
                 seed: int | None = None,
                 header_row: bool = True):
        self.path_to_training_data = path_to_training_data
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.output_directory = output_directory
        self.output_file_prefix = output_file_prefix

        self.next_yield_indexes: List[int] = []

        # Validate the file exists
        if not self.path_to_training_data.exists():
            raise FileNotFoundError(f"Training data file not found: {self.path_to_training_data}")

        # Validate the file is a CSV
        if not self.path_to_training_data.suffix == '.csv':
            raise ValueError(f"Training data file is not a CSV: {self.path_to_training_data}")

        # Validate the file is not empty
        if self.path_to_training_data.stat().st_size == 0:
            raise ValueError(f"Training data file is empty: {self.path_to_training_data}")

        self.has_header_row = header_row
        if self.has_header_row:
            # Read the header row from the file
            self.header_row = pd.read_csv(self.path_to_training_data, nrows=1).columns.tolist()
        else:
            self.header_row = None

    def make_splits(self,
                    split_ratios: List[float] = [8, 1, 1],
                    delete_old_splits: bool = True,
                    shuffle: bool = True,
                    batch_size: int = 32,
                    load_chunk_size: int = 10_000):
        """Stream the source CSV in chunks, split each chunk by ratio, append to split files.

        pandas CSV rules this method relies on:
        - `chunksize=` makes `read_csv` return an iterator of DataFrames, not one DataFrame.
        - `header=0` (the default): row 0 is column names, remaining rows are data.
        - `header=None`: every row is data; columns are integers 0, 1, 2, ...
        - Do not `next(file)` the header yourself. pandas already consumes it when `header=0`.
        - A DataFrame slice is still a DataFrame. Write it with `to_csv`; do not pass it to
          `csv.writer.writerows` (that iterates column *names*, not rows).
        """
        splits = TTVSplits.from_unnormalized(split_ratios)
        output_paths = self._prepare_split_output_files(
            splits.num_splits,
            delete_old_splits
        )
        file_iterators: List[TextFileReader] = []

        reader = pd.read_csv(
            self.path_to_training_data,
            chunksize=load_chunk_size,
            header=0 if self.has_header_row else None,
        )

        chunk: pd.DataFrame = None  # For type hinting
        for chunk in reader:
            if shuffle:
                chunk = chunk.sample(
                    frac=1,  # Keep every row
                    random_state=self.rng.randint(0, MAX_INT),  # Shuffle the chunk
                ).reset_index(drop=True)

            for i, (start_index, end_index) in enumerate(splits.get_split_indices(len(chunk))):
                # Start is inclusive, end is exclusive
                if start_index == end_index:
                    continue
                slice_df: pd.DataFrame = chunk.iloc[start_index:end_index]
                slice_df.to_csv(
                    output_paths[i],
                    mode="a",
                    index=False,   # Omit the DataFrame row index (0,1,2,...) from the file.
                    header=False,  # Header was written by `_prepare_split_output_files`.
                )
            # Finished writing the chunk to the split files
        # Finished streaming the data from the source file

        for output_path in output_paths:
            file_iterators.append(
                pd.read_csv(output_path, iterator=True, chunksize=batch_size)
            )
        return file_iterators

    def shuffle_split(self, split: Split,
                      batch_size: int = 32):
        if split == Split.TRAIN:
            return self.shuffle_split_index(0, batch_size)
        elif split == Split.TEST:
            return self.shuffle_split_index(1, batch_size)
        elif split == Split.VALIDATION:
            return self.shuffle_split_index(2, batch_size)
        else:
            raise ValueError(f"Unsupported split: {split}")

    def shuffle_split_index(self, split_index: int,
                            batch_size: int = 32) -> TextFileReader:
        target_file = self.output_directory / \
            f"{self.output_file_prefix}_{self._get_file_postfix(split_index)}.csv"
        file = pd.read_csv(target_file)
        file = file.sample(
            frac=1,
            random_state=self.rng.randint(0, MAX_INT)
        ).reset_index(drop=True)
        # TODO: Will this overwrite the file? Or append to it?
        file.to_csv(target_file, index=False)
        return pd.read_csv(target_file, iterator=True, chunksize=batch_size)

    def _prepare_split_output_files(self, num_splits: int,
                                    delete_old_splits: bool = True) -> List[Path]:
        """Create/truncate each split CSV and write the header row if the source has one."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if delete_old_splits:
            for path in self.output_directory.glob(f"{self.output_file_prefix}*.csv"):
                path.unlink()
        output_paths = [
            self.output_directory / f"{self.output_file_prefix}_{self._get_file_postfix(i)}.csv"
            for i in range(num_splits)
        ]
        header_frame = pd.DataFrame(columns=self.header_row if self.has_header_row else [])
        for path in output_paths:
            header_frame.to_csv(path, index=False, header=self.has_header_row)
        return output_paths

    def _get_file_postfix(self, split_index: int) -> str:
        if split_index == 0:
            return "train"
        elif split_index == 1:
            return "test"
        elif split_index == 2:
            return "validation"
        else:
            return f"split_{split_index}"
