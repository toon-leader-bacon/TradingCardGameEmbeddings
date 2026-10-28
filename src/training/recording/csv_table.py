"""CsvTable: one CSV file written row by row, for recorders whose rows
must survive a crash (the run sampler)."""

import csv
import threading
from pathlib import Path
from typing import Mapping


class CsvTable:
    """A CSV with a fixed header, written under a lock and flushed every
    row; thread-safe. Floats are written to 6 decimals."""

    def __init__(self, path: Path, fields: list[str]) -> None:
        """Inputs: path (must not exist; its directory must), fields (the
            header, in order).
        Output: none. Side effects: creates the file and writes the header.
        Exceptions: FileExistsError if path exists; OSError.

        Example:
            >>> table = CsvTable(directory / "steps.csv", ["t", "dojo"])
        """
        self._fields = list(fields)
        self._lock = threading.Lock()
        self._handle = path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self._fields)
        self._writer.writeheader()
        self._handle.flush()
        self._closed = False

    @property
    def fields(self) -> list[str]:
        """The header, in order. Inputs: none. Output: list[str] (a copy).
        Side effects: none. Exceptions: none."""
        return list(self._fields)

    def write(self, row: Mapping[str, object]) -> None:
        """Write one row and flush; a closed table drops it.

        Inputs: row (keys among fields; a missing key is written empty).
        Output: none. Side effects: appends to the file.
        Exceptions: ValueError for a key not in fields; OSError.

        Example:
            >>> table.write({"t": 1.5, "dojo": "contrastive.mtg"})
        """
        formatted = {
            key: f"{value:.6f}" if isinstance(value, float) else value
            for key, value in row.items()
        }
        with self._lock:
            if self._closed:
                return
            self._writer.writerow(formatted)
            self._handle.flush()

    def close(self) -> None:
        """Close the file; idempotent. Inputs: none. Output: none.
        Side effects: closes the handle. Exceptions: none.

        Example:
            >>> table.close()
        """
        with self._lock:
            if not self._closed:
                self._closed = True
                self._handle.close()
