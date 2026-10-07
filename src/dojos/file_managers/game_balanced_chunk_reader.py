"""GameBalancedChunkReader: a split file's rows, drawn evenly across the
values of one column.

A cross-game dojo's TRAIN split is dominated by its biggest game (MTG has
35k cards, StS2 under 600). Reading the split in file order would train
almost entirely on MTG. This reader draws each row by first picking a
column value (a game) uniformly, then one of that value's rows uniformly,
with replacement. A game with few rows is therefore repeated; no game is
dropped.
"""

import random
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd


class GameBalancedChunkReader:
    """One pass over a split file, rows drawn evenly across a column.

    A pass yields as many rows as pass keep_row, in chunks of chunk_rows
    (so an epoch costs about the same as an unbalanced one). The rows a
    consumer will discard anyway are removed first by keep_row: balancing
    before they are dropped would skew the draw toward the games with
    fewer of them. The split file is read into memory when a pass starts;
    it is meant for TRAIN splits of tens of thousands of rows, not the
    100M-row ones.

    Inputs (constructor): path (a split parquet file), balance_column
        (e.g. "source_game"), chunk_rows (rows per yielded chunk), rng
        (shared across passes, so each pass draws differently), keep_row
        (a frame -> boolean mask of the rows that may be drawn; default
        keeps every row).
    Output: n/a.
    Side effects: none until iterated.
    Exceptions: none from the constructor.

    Example:
        >>> reader = GameBalancedChunkReader(
        ...     Path("data/splits/x_train.parquet"), "source_game", 256,
        ...     random.Random(0))
        >>> sum(len(chunk) for chunk in reader) == rows_in_the_file
        True
    """

    def __init__(
        self,
        path: Path,
        balance_column: str,
        chunk_rows: int,
        rng: random.Random,
        keep_row: Callable[[pd.DataFrame], pd.Series] = lambda frame: pd.Series(
            True, index=frame.index
        ),
    ) -> None:
        self._path = path
        self._balance_column = balance_column
        self._chunk_rows = chunk_rows
        self._rng = rng
        self._keep_row = keep_row

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Draw one pass's rows and yield them in chunks.

        Output: DataFrames with the file's columns (pandas ArrowDtype, as
            ParquetChunkReader yields), at most chunk_rows rows each;
            their row counts sum to the number of rows keep_row accepts.
        Side effects: reads the file; advances the shared rng.
        Exceptions: FileNotFoundError for a missing file; KeyError if the
            file lacks balance_column.
        """
        table = pd.read_parquet(self._path, dtype_backend="pyarrow")
        table = table[self._keep_row(table)].reset_index(drop=True)
        rows_by_value = _row_positions_by_value(table[self._balance_column])
        values = sorted(rows_by_value)

        # Draw every row of the pass: a value uniformly, then one of its rows
        draws: list[int] = []
        for _ in range(len(table)):
            value = self._rng.choice(values)
            draws.append(self._rng.choice(rows_by_value[value]))

        # Hand the draws out chunk by chunk
        for start in range(0, len(draws), self._chunk_rows):
            yield table.iloc[draws[start : start + self._chunk_rows]]


def _row_positions_by_value(column: pd.Series) -> dict[str, list[int]]:
    """Each distinct value of column -> the row positions that hold it.

    Inputs: column (one string column). Output: dict value -> positions, in
    file order. Side effects: none. Exceptions: none.
    """
    result: dict[str, list[int]] = {}
    for position, value in enumerate(column):
        result.setdefault(str(value), []).append(position)
    return result
