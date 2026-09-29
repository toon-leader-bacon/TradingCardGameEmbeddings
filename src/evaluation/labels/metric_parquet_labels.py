"""MetricParquetLabels: card labels read from per-card categorical metric
parquets, the way dojos read them (nocab_uuid + label columns only)."""

import math
from pathlib import Path
from typing import Sequence
from uuid import UUID

import pyarrow.parquet as pq

from src.evaluation.card_row import CardRow

# The metric-parquet convention these labels read (plans/evaluation.md has
# an open question on giving it a shared home)
_UUID_COLUMN = "nocab_uuid"
_LABEL_COLUMN = "label"


class MetricParquetLabels:
    """Labels from one or more per-card categorical metric parquets.

    Reads only the nocab_uuid and label columns (e.g. MaskedFieldMetric's
    nocab_uuid / masked_field / label), converting labels with str().
    Several parquets form one label source, so a per-game metric family
    (one parquet per game) labels a multi-game corpus. A row with a null
    label leaves its card unlabeled. Unlike GenericDojo, the metric's
    binder version is not checked against the current binder: nocab_uuids
    are stable.

    Everything is read and checked in the constructor, so a bad file fails
    before any analysis runs.
    """

    def __init__(self, name: str, parquet_paths: Sequence[Path]) -> None:
        """Inputs: name (non-empty), parquet_paths (non-empty).
        Output: none (constructor).
        Side effects: reads every parquet.
        Exceptions: ValueError if name or parquet_paths is empty, a
            parquet lacks the nocab_uuid or label column, a nocab_uuid does
            not parse, or a nocab_uuid appears in more than one row across
            all the files (e.g. a per-deck metric; the message names the
            file); FileNotFoundError for a missing file.

        Example:
            >>> labels = MetricParquetLabels(
            ...     "set", [Path("data/metrics/dominiontabs/set_mask.parquet")])
            >>> labels.label_of(row)
            'adventures'
        """
        self.name = name
        self._labels: dict[UUID, str] = {}
        _require_non_empty(name, parquet_paths)

        # Merge every file's rows, refusing a card that appears twice (even
        # when one of the rows has a null label); keep only real labels
        seen: set[UUID] = set()
        for path in parquet_paths:
            for nocab_uuid, label in _read_labels(path):
                if nocab_uuid in seen:
                    raise ValueError(f"{path}: card {nocab_uuid} appears twice")
                seen.add(nocab_uuid)
                if label is not None:
                    self._labels[nocab_uuid] = label

    def label_of(self, row: CardRow) -> str | None:
        """Inputs: row (CardRow). Output: its label, or None if no parquet
        labels it. Side effects: none. Exceptions: none.

        Example:
            >>> labels.label_of(CardRow(uuid4(), GameId.MTG)) is None
            True
        """
        return self._labels.get(row.nocab_uuid)


def _require_non_empty(name: str, parquet_paths: Sequence[Path]) -> None:
    """Inputs: a label source's name and paths. Output: None. Side
    effects: none. Exceptions: ValueError if name or parquet_paths is
    empty."""
    if not name:
        raise ValueError("a label source needs a non-empty name")
    if not parquet_paths:
        raise ValueError(f"label source {name!r} needs at least one parquet")


def _read_labels(path: Path) -> list[tuple[UUID, str | None]]:
    """One parquet's (nocab_uuid, label) pairs, one per row, in file order.

    Inputs: path (Path). Output: list[tuple[UUID, str | None]]: labels via
        str(), a null label as None (kept, so the caller's duplicate check
        sees every row).
    Side effects: reads the file (only its nocab_uuid and label columns).
    Exceptions: FileNotFoundError if absent or not a file (a partitioned
        dataset directory is not supported); ValueError naming the file if
        either column is missing or a nocab_uuid does not parse.
    """
    result: list[tuple[UUID, str | None]] = []
    # A single file only; pyarrow's own missing-file error type varies
    if not path.is_file():
        raise FileNotFoundError(path)
    missing = {_UUID_COLUMN, _LABEL_COLUMN} - set(pq.read_schema(path).names)
    if missing:
        raise ValueError(f"{path} lacks column(s) {sorted(missing)}")

    # Read the two columns with pyarrow, not pandas: pandas turns an int
    # column holding a null into floats, so 3 would become the label "3.0"
    table = pq.read_table(path, columns=[_UUID_COLUMN, _LABEL_COLUMN])
    uuid_cells = table.column(_UUID_COLUMN).to_pylist()
    label_cells = table.column(_LABEL_COLUMN).to_pylist()
    for uuid_cell, label_cell in zip(uuid_cells, label_cells):
        result.append((_parse_uuid(path, uuid_cell), _to_label_text(label_cell)))
    return result


def _parse_uuid(path: Path, value: object) -> UUID:
    """Inputs: the file (for messages) and one nocab_uuid cell. Output:
    UUID. Side effects: none. Exceptions: ValueError naming the file if
    the cell is not a UUID string."""
    try:
        return UUID(str(value))
    except ValueError as error:
        raise ValueError(f"{path}: bad nocab_uuid {value!r}") from error


def _to_label_text(value: object) -> str | None:
    """Inputs: one label cell as a Python value (pyarrow's to_pylist).
    Output: str(value), or None for a null cell or a float NaN.
    Side effects: none. Exceptions: none."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)
