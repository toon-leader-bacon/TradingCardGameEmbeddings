"""
Scans a CSV file, breaks it into chunks, and passes those chunks into
a collection of metrics. Finalize will be called on each metric after the
CSV has been completely scanned.
"""

import warnings
from pathlib import Path

import pandas as pd

from src.data_refinement.seventeenlands.metric import Metric


class CsvScanner:
    """Streams raw_csv_path in chunks, driving every metric in metrics
    over each chunk, then finalizing every metric once the file is
    exhausted.

    Single-consumer-per-scan — one CsvScanner instance handles one
    (raw_csv_path, metrics) combination; a different CSV or a
    different set of active metrics gets its own instance.
    """

    def __init__(
        self, raw_csv_path: Path, chunk_size: int, metrics: list[Metric]
    ) -> None:
        """
        Inputs:
            raw_csv_path: path to the raw CSV to stream.
            chunk_size: how many rows per chunk, passed straight
                through to pandas.read_csv's chunksize.
            metrics: every Metric to drive over this CSV in one pass.
                Each must already be fully constructed (e.g. already
                knows its own expansion/format_code/card_binder — that
                wiring happens before this class ever sees a metric,
                not here).
        Output: none (constructor).
        Side effects: none — no I/O happens until scan() is called.
        Exceptions: none.
        """
        self.raw_csv_path = raw_csv_path
        self.chunk_size = chunk_size
        self.metrics = metrics

    def scan(self) -> list[Path]:
        """Stream raw_csv_path in chunks, driving every metric, then
        finalize every metric and report where each one wrote.

        Composed of:
            1. Stream raw_csv_path via
               pandas.read_csv(chunksize=self.chunk_size).
            2. For each chunk, call metric.accumulate(chunk) for every
               metric in self.metrics, in order.
            3. Once every chunk has been seen, call metric.finalize()
               for every metric in self.metrics, in order, collecting
               each returned Path.
            4. For each collected Path, warn (do not raise) if it
               doesn't exist — a metric reporting a path that was
               never actually written is a bug in that metric, worth
               surfacing, but not grounds for failing every other
               active metric's already-successful output.

        Inputs: none (uses constructor-supplied state).
        Output: every metric's finalize() return value, in the same
            order as self.metrics.
        Side effects: reads raw_csv_path; drives every metric's own
            accumulate()/finalize() side effects (typically writing to
            each metric's own output file).
        Exceptions: raises on failure to read raw_csv_path, or
            whatever a metric's own accumulate()/finalize() raises.

        Example:
            >>> scanner = CsvScanner(
            ...     Path("data/raw/17lands/game_data/MSH.PremierDraft.csv"),
            ...     chunk_size=100_000,
            ...     metrics=[deck_outcome_metric, win_rate_metric],
            ... )
            >>> output_paths = scanner.scan()
        """
        for chunk in pd.read_csv(self.raw_csv_path, chunksize=self.chunk_size):
            for metric in self.metrics:
                metric.accumulate(chunk)

        output_paths = [metric.finalize() for metric in self.metrics]
        for metric, output_path in zip(self.metrics, output_paths):
            if not output_path.exists():
                warnings.warn(
                    f"Metric {metric.name!r} finalize() reported {output_path}, "
                    "but that path doesn't exist",
                    RuntimeWarning,
                )
        return output_paths
