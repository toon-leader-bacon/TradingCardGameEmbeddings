"""TeacherNeighbours: each teacher row's neighbours, by cosine, as a
NeighbourRule picks them. Ragged (CSR), since a threshold rule gives each
card a different count.

Only confident rows take part, as anchors and as neighbours: a row that
is not confident has no neighbours and is no one's neighbour. A
confident row whose vector is all zero (its decks held only dropped
cards) has cosine 0 to every row: a top-K rule still gives it
neighbours, ranked arbitrarily; a cosine threshold above 0 gives it
none. On disk:
one single-batch Arrow file, one row per teacher row, with list<int32>
neighbour_rows and list<float32> neighbour_cosines columns (an Arrow list
column is already CSR, so offsets and values map straight to NumPy).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from src.data_refinement.snapshots.arrow_files import (
    open_mapped_table,
    write_table_atomically,
)
from src.teacher.neighbour_rules import NeighbourRule
from src.teacher.teacher_table import (
    StaleTeacherError,
    TeacherStamp,
    TeacherTable,
    require_metadata_value,
)

_logger = logging.getLogger(__name__)

# Teacher rows whose similarities are computed in one matrix product
# (peak ~0.5 GB for 1024 x 30k MTG rows: the similarities, their
# negation and argpartition's int64 output)
_ROWS_PER_BLOCK = 1_024

# Bumped when the neighbours file's columns or their meaning change
_NEIGHBOURS_FORMAT_VERSION = 1

# The file's columns and metadata keys (beside the teacher stamp's)
_ROWS_COLUMN = "neighbour_rows"
_COSINES_COLUMN = "neighbour_cosines"
_RULE_KEY = b"neighbour_rule"
_FORMAT_VERSION_KEY = b"teacher_neighbours_format_version"


@dataclass(frozen=True, eq=False)
class TeacherNeighbours:
    """Each teacher row's neighbours.

    teacher_stamp: the table these rows index. rule: how they were picked.
    offsets: int64 (cards + 1,); row r's neighbours are
        indices[offsets[r]:offsets[r + 1]].
    indices: int32, teacher rows, cosine-descending within each row.
    cosines: float32, aligned with indices.
    """

    teacher_stamp: TeacherStamp
    rule: NeighbourRule
    offsets: np.ndarray
    indices: np.ndarray
    cosines: np.ndarray

    def __post_init__(self) -> None:
        """Raise ValueError unless the CSR arrays agree."""
        # Validate inputs: the CSR arrays agree with each other
        if self.offsets.ndim != 1 or len(self.offsets) == 0 or self.offsets[0] != 0:
            raise ValueError("offsets must be 1-D and start at 0")
        if not int(self.offsets[-1]) == len(self.indices) == len(self.cosines):
            raise ValueError(
                f"offsets end at {int(self.offsets[-1])}, but there are "
                f"{len(self.indices)} indices and {len(self.cosines)} cosines"
            )

    @property
    def row_count(self) -> int:
        """Teacher rows covered (len(offsets) - 1).

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return len(self.offsets) - 1

    def neighbours_of(self, row: int) -> np.ndarray:
        """row's neighbour rows, most similar first.

        Inputs: row (a teacher row). Output: np.ndarray int32 (a view).
        Side effects: none. Exceptions: IndexError if row is out of range.

        Example:
            >>> neighbours.neighbours_of(0)[:3]
            array([812, 77, 1203], dtype=int32)
        """
        if not 0 <= row < self.row_count:
            raise IndexError(f"row {row} is not in 0..{self.row_count - 1}")
        return self.indices[self.offsets[row] : self.offsets[row + 1]]

    def save(self, path: Path) -> None:
        """Write to path (atomically; layout in the module docstring), with
        the teacher stamp and the rule's payload in the schema metadata.

        Inputs: path. Output: none.
        Side effects: creates path's parent; writes path.
        Exceptions: OSError on a write failure.

        Example:
            >>> neighbours.save(directory / "3f2b1c0a9d8e.a1b2c3d4e5f6.neighbours.arrow")
        """
        offsets = pa.array(self.offsets.astype(np.int32))
        columns = {
            _ROWS_COLUMN: pa.ListArray.from_arrays(offsets, pa.array(self.indices)),
            _COSINES_COLUMN: pa.ListArray.from_arrays(offsets, pa.array(self.cosines)),
        }
        metadata = {
            **self.teacher_stamp.to_schema_metadata(),
            _RULE_KEY: _rule_text(self.rule).encode(),
            _FORMAT_VERSION_KEY: str(_NEIGHBOURS_FORMAT_VERSION).encode(),
        }
        table = pa.table(columns).replace_schema_metadata(metadata)
        write_table_atomically(table, path)

    @classmethod
    def open(cls, path: Path, rule: NeighbourRule) -> "TeacherNeighbours":
        """Memory-map a file save() wrote for rule.

        Inputs: path, rule (the file must record rule's payload).
        Output: TeacherNeighbours (read-only views of the file).
        Side effects: maps the file.
        Exceptions: FileNotFoundError if missing; StaleTeacherError if
            malformed, of another format, or written for another rule.

        Example:
            >>> TeacherNeighbours.open(path, TopKNeighbours(32)).teacher_stamp == stamp
            True
        """
        table = open_mapped_table(path)
        metadata = table.schema.metadata

        # Validate: this format, written for this rule
        require_metadata_value(
            metadata, _FORMAT_VERSION_KEY, str(_NEIGHBOURS_FORMAT_VERSION), path
        )
        require_metadata_value(metadata, _RULE_KEY, _rule_text(rule), path)

        # The list columns are CSR already
        rows = _only_chunk(table.column(_ROWS_COLUMN), path)
        cosines = _only_chunk(table.column(_COSINES_COLUMN), path)
        return cls(
            teacher_stamp=TeacherStamp.from_schema_metadata(metadata),
            rule=rule,
            offsets=rows.offsets.to_numpy().astype(np.int64),
            indices=rows.values.to_numpy(),
            cosines=cosines.values.to_numpy(),
        )


def find_neighbours(table: TeacherTable, rule: NeighbourRule) -> TeacherNeighbours:
    """Every confident row's neighbours among the other confident rows, by
    cosine of the teacher vectors, as rule picks them.

    Inputs: table, rule.
    Output: TeacherNeighbours over table's rows (non-confident rows: none).
    Side effects: logs progress.
    Exceptions: none.

    Example:
        >>> find_neighbours(table, TopKNeighbours(32)).neighbours_of(0)
    """
    unit_vectors = _unit_rows(table.vectors)
    confident_rows = np.flatnonzero(table.confident)
    candidate_vectors = unit_vectors[confident_rows]
    kept_per_row = np.zeros(len(table.card_uuids), dtype=np.int64)
    kept_indices: list[np.ndarray] = []
    kept_cosines: list[np.ndarray] = []

    # Rank each block of confident rows against every confident row
    for start in range(0, len(confident_rows), _ROWS_PER_BLOCK):
        block = confident_rows[start : start + _ROWS_PER_BLOCK]
        ranked_rows, ranked_cosines = _ranked_candidates(
            unit_vectors[block],
            block,
            candidate_vectors,
            confident_rows,
            rule.candidate_limit,
        )
        # Let the rule cut each row's ranked candidates
        for position, row in enumerate(block):
            kept = rule.kept_count(ranked_cosines[position])
            kept_per_row[row] = kept
            kept_indices.append(ranked_rows[position, :kept])
            kept_cosines.append(ranked_cosines[position, :kept])

    # Rows were visited ascending, so the kept slices are already in CSR order
    offsets = np.concatenate([[0], np.cumsum(kept_per_row)]).astype(np.int64)
    result = TeacherNeighbours(
        teacher_stamp=table.stamp,
        rule=rule,
        offsets=offsets,
        indices=_concatenated(kept_indices, np.int32),
        cosines=_concatenated(kept_cosines, np.float32),
    )
    return result


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """vectors L2-normalised row by row (float32); a zero row stays zero.

    Private helper - single caller is find_neighbours.
    Inputs: vectors (np.ndarray float32, 2-D).
    Output: np.ndarray float32, same shape. Side effects: none.
    Exceptions: none.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return (vectors / safe_norms).astype(np.float32)


def _ranked_candidates(
    block_vectors: np.ndarray,
    block: np.ndarray,
    candidate_vectors: np.ndarray,
    confident_rows: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each row of block, its up-to-limit most similar confident rows
    other than itself, cosine-descending (argpartition, then sort the
    kept slice). Rows with fewer candidates are padded with cosine -inf,
    which no rule keeps.

    Private helper - single caller is find_neighbours.
    Inputs: unit_vectors (np.ndarray float32), block (np.ndarray int64,
        teacher rows), confident_rows (np.ndarray int64), limit (int).
    Output: (int32 (len(block), limit) teacher rows, float32 cosines).
    Side effects: none. Exceptions: none.
    """
    similarities = block_vectors @ candidate_vectors.T
    # A row is never its own neighbour
    is_self = block[:, None] == confident_rows[None, :]
    similarities[is_self] = -np.inf

    # The top `taken` per row, unordered, then sorted descending
    taken = min(limit, len(confident_rows))
    top = np.argpartition(-similarities, taken - 1, axis=1)[:, :taken]
    top_cosines = np.take_along_axis(similarities, top, axis=1)
    order = np.argsort(-top_cosines, axis=1, kind="stable")
    ranked_positions = np.take_along_axis(top, order, axis=1)
    ranked_cosines = np.take_along_axis(top_cosines, order, axis=1)

    # Pad to limit with -inf, which no rule keeps
    result_rows = np.full((len(block), limit), -1, dtype=np.int32)
    result_cosines = np.full((len(block), limit), -np.inf, dtype=np.float32)
    result_rows[:, :taken] = confident_rows[ranked_positions]
    result_cosines[:, :taken] = ranked_cosines
    return result_rows, result_cosines


def _concatenated(slices: list[np.ndarray], dtype: type) -> np.ndarray:
    """slices joined into one array of dtype (empty when slices is).

    Private helper - callers are find_neighbours (indices and cosines).
    Inputs: slices (list[np.ndarray]), dtype (a NumPy scalar type).
    Output: np.ndarray. Side effects: none.
    Exceptions: none.
    """
    if not slices:
        return np.empty(0, dtype=dtype)
    return np.concatenate(slices).astype(dtype, copy=False)


def _rule_text(rule: NeighbourRule) -> str:
    """rule's payload as canonical JSON (what a file records and is
    checked against).

    Private helper - callers are TeacherNeighbours.save and open.
    Inputs: rule (NeighbourRule). Output: str. Side effects: none.
    Exceptions: none.
    """
    return json.dumps(rule.payload(), sort_keys=True)


def _only_chunk(column: pa.ChunkedArray, path: Path) -> pa.Array:
    """column's one chunk.

    Private helper - single caller is TeacherNeighbours.open.
    Inputs: column (pa.ChunkedArray), path (Path, for the message).
    Output: pa.Array. Side effects: none.
    Exceptions: StaleTeacherError if column has another chunk count.
    """
    if column.num_chunks != 1:
        raise StaleTeacherError(
            f"{path}: a column has {column.num_chunks} chunks, not 1"
        )
    return column.chunk(0)
