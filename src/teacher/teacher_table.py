"""TeacherTable: one game's teacher vectors, the only contract between
the teacher and the student, and the TeacherStamp naming everything it
was built from.

On disk: one single-batch Arrow file (snapshots/arrow_files.py), one row
per selected card: nocab_uuid fixed_size_binary(16), vector
fixed_size_list<float32>[dim], deck_count int64, confident bool. The
stamp and the game ride in the schema metadata.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np
import pyarrow as pa

from src.data_refinement.snapshots.arrow_files import (
    fixed_width_bytes_of,
    open_mapped_table,
    write_table_atomically,
)
from src.data_refinement.snapshots.columns import UUID_WIDTH
from src.data_refinement.snapshots.stamps import DeckSnapshotStamp, StaleSnapshotError
from src.schema.game_id import GameId

# Hex digits of a stamp's sha256 in a cache file's name
_STAMP_TAG_LENGTH = 12

# Bumped when the table file's columns or their meaning change
TEACHER_TABLE_FORMAT_VERSION = 1

# Schema metadata keys, beside the deck snapshot stamp's own
_TEACHER_STAMP_KEY = b"teacher_stamp"
_SOURCE_GAME_KEY = b"source_game"
_FORMAT_VERSION_KEY = b"teacher_table_format_version"


class StaleTeacherError(Exception):
    """A teacher file is unreadable, of another format version, or holds
    another stamp than the one asked for."""


@dataclass(frozen=True)
class TeacherStamp:
    """Everything a fitted TeacherTable depends on.

    deck_snapshot: the snapshot counted (nests the CardSnapshotStamp,
        and so the binder version).
    deck_split_key: holdout.deck_split_key(game): which decks are which split.
    card_tier_key: holdout.card_tier_key(): which cards are which tier.
    fit_digest: TeacherSettings.fit_digest(): splits, tiers, counting
        rules, method and min_deck_count.
    """

    deck_snapshot: DeckSnapshotStamp
    deck_split_key: str
    card_tier_key: str
    fit_digest: str

    def tag(self) -> str:
        """A short, file-name-safe tag of this stamp.

        Inputs: none. Output: str (hex). Side effects: none.
        Exceptions: none.

        Example:
            >>> stamp.tag()
            '3f2b1c0a9d8e'
        """
        return stamp_tag_of(dataclasses.asdict(self))

    def to_schema_metadata(self) -> dict[bytes, bytes]:
        """This stamp as Arrow schema metadata: the deck snapshot stamp's
        keys plus one JSON entry for the rest.

        Inputs: none. Output: dict[bytes, bytes].
        Side effects: none. Exceptions: none.

        Example:
            >>> schema.with_metadata(stamp.to_schema_metadata())
        """
        own_fields = {
            "deck_split_key": self.deck_split_key,
            "card_tier_key": self.card_tier_key,
            "fit_digest": self.fit_digest,
        }
        return {
            **self.deck_snapshot.to_schema_metadata(),
            _TEACHER_STAMP_KEY: json.dumps(own_fields, sort_keys=True).encode(),
        }

    @classmethod
    def from_schema_metadata(
        cls, metadata: dict[bytes, bytes] | None
    ) -> "TeacherStamp":
        """Read a stamp back from Arrow schema metadata.

        Inputs: metadata (a file's schema.metadata; None when it has none).
        Output: TeacherStamp.
        Side effects: none.
        Exceptions: StaleTeacherError if a key is missing or unreadable
            (the deck snapshot stamp's StaleSnapshotError is re-raised as
            one, so callers handle a single error).

        Example:
            >>> TeacherStamp.from_schema_metadata(table.schema.metadata)
        """
        try:
            deck_snapshot = DeckSnapshotStamp.from_schema_metadata(metadata)
        except StaleSnapshotError as error:
            raise StaleTeacherError(
                f"teacher file has no deck stamp: {error}"
            ) from error
        fields = _stamp_fields_of(metadata)
        return cls(
            deck_snapshot=deck_snapshot,
            deck_split_key=fields["deck_split_key"],
            card_tier_key=fields["card_tier_key"],
            fit_digest=fields["fit_digest"],
        )


@dataclass(frozen=True, eq=False)
class TeacherTable:
    """One game's teacher: a vector per selected card.

    source_game: the game. card_uuids: the selected cards; row i of every
    array is card_uuids[i]. vectors: float32 (cards, dim). deck_counts:
    int64 (cards,), selected decks holding each card. confident: bool
    (cards,), deck_counts >= the settings' min_deck_count. stamp: what it
    was built from.
    """

    source_game: GameId
    card_uuids: tuple[UUID, ...]
    vectors: np.ndarray
    deck_counts: np.ndarray
    confident: np.ndarray
    stamp: TeacherStamp

    def __post_init__(self) -> None:
        """Raise ValueError unless every array has one row per card."""
        # Validate inputs: every array has one row per card
        row_count = len(self.card_uuids)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != row_count:
            raise ValueError(
                f"vectors has shape {self.vectors.shape}, want ({row_count}, dim)"
            )
        if self.deck_counts.shape != (row_count,) or self.confident.shape != (
            row_count,
        ):
            raise ValueError("deck_counts and confident need one entry per card")

    @property
    def dim(self) -> int:
        """The teacher vector width.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return int(self.vectors.shape[1])

    def save(self, path: Path) -> None:
        """Write this table to path (atomically; see the module docstring
        for the layout).

        Inputs: path. Output: none.
        Side effects: creates path's parent; writes path.
        Exceptions: OSError on a write failure.

        Example:
            >>> table.save(directory / f"{table.stamp.tag()}.teacher.arrow")
        """
        columns = {
            "nocab_uuid": _uuid_column(self.card_uuids),
            "vector": _vector_column(self.vectors),
            "deck_count": pa.array(self.deck_counts, type=pa.int64()),
            "confident": pa.array(self.confident, type=pa.bool_()),
        }
        metadata = {
            **self.stamp.to_schema_metadata(),
            _SOURCE_GAME_KEY: self.source_game.value.encode(),
            _FORMAT_VERSION_KEY: str(TEACHER_TABLE_FORMAT_VERSION).encode(),
        }
        table = pa.table(columns).replace_schema_metadata(metadata)
        write_table_atomically(table, path)

    @classmethod
    def open(cls, path: Path) -> "TeacherTable":
        """Memory-map a table save() wrote.

        Inputs: path.
        Output: TeacherTable (its arrays are read-only views of the file).
        Side effects: maps the file.
        Exceptions: FileNotFoundError if path is missing; StaleTeacherError
            if the file is malformed or of another format version.

        Example:
            >>> TeacherTable.open(path).stamp == expected_stamp
        """
        table = open_mapped_table(path)
        metadata = table.schema.metadata

        # Validate: this format, then read the stamp and the columns
        require_metadata_value(
            metadata, _FORMAT_VERSION_KEY, str(TEACHER_TABLE_FORMAT_VERSION), path
        )
        return cls(
            source_game=_source_game_of(metadata, path),
            card_uuids=_uuids_of(table.column("nocab_uuid")),
            vectors=_vectors_of(table.column("vector")),
            deck_counts=table.column("deck_count").to_numpy(),
            confident=table.column("confident").to_numpy(zero_copy_only=False),
            stamp=TeacherStamp.from_schema_metadata(metadata),
        )


def _stamp_fields_of(metadata: dict[bytes, bytes] | None) -> dict[str, str]:
    """The teacher stamp's own JSON entry, as its three string fields.

    Private helper - single caller is TeacherStamp.from_schema_metadata.
    Inputs: metadata (dict[bytes, bytes] | None).
    Output: dict[str, str] with deck_split_key, card_tier_key, fit_digest.
    Side effects: none.
    Exceptions: StaleTeacherError if the entry is missing, not JSON, or
        lacks a field.
    """
    raw = (metadata or {}).get(_TEACHER_STAMP_KEY)
    if raw is None:
        raise StaleTeacherError("teacher file has no teacher stamp")
    try:
        fields = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StaleTeacherError("teacher stamp is not JSON") from error
    wanted = ("deck_split_key", "card_tier_key", "fit_digest")
    if not isinstance(fields, dict) or not all(
        isinstance(fields.get(name), str) for name in wanted
    ):
        raise StaleTeacherError(f"teacher stamp lacks one of {wanted}")
    return {name: fields[name] for name in wanted}


def _uuid_column(card_uuids: tuple[UUID, ...]) -> pa.Array:
    """card_uuids as a fixed_size_binary(16) array.

    Private helper - single caller is TeacherTable.save.
    Inputs: card_uuids. Output: pa.Array. Side effects: none.
    Exceptions: none.
    """
    flat = b"".join(card_uuid.bytes for card_uuid in card_uuids)
    return pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(UUID_WIDTH), len(card_uuids), [None, pa.py_buffer(flat)]
    )


def _vector_column(vectors: np.ndarray) -> pa.Array:
    """vectors as a fixed_size_list<float32>[dim] array (one buffer).

    Private helper - single caller is TeacherTable.save.
    Inputs: vectors (float32, 2-D). Output: pa.Array. Side effects: none.
    Exceptions: none.
    """
    values = pa.array(np.ascontiguousarray(vectors, dtype=np.float32).ravel())
    return pa.FixedSizeListArray.from_arrays(values, vectors.shape[1])


def require_metadata_value(
    metadata: dict[bytes, bytes] | None, key: bytes, expected: str, path: Path
) -> None:
    """Raise StaleTeacherError unless metadata[key] decodes to expected:
    the one check of a teacher file's format version or rule.

    Inputs: metadata (dict[bytes, bytes] | None), key (bytes), expected
        (str), path (Path, for the message).
    Output: None. Side effects: none.
    Exceptions: StaleTeacherError naming path and key.

    Example:
        >>> require_metadata_value(table.schema.metadata, b"rule", '{"k": 32}', path)
    """
    raw = (metadata or {}).get(key)
    if raw is None or raw.decode(errors="replace") != expected:
        raise StaleTeacherError(f"{path}: {key.decode()} is not {expected!r}")


def _source_game_of(metadata: dict[bytes, bytes] | None, path: Path) -> GameId:
    """The game recorded in metadata.

    Private helper - single caller is TeacherTable.open.
    Inputs: metadata (dict[bytes, bytes] | None), path (Path).
    Output: GameId. Side effects: none.
    Exceptions: StaleTeacherError if missing or unknown.
    """
    raw = (metadata or {}).get(_SOURCE_GAME_KEY)
    if raw is None:
        raise StaleTeacherError(f"{path} records no game")
    try:
        return GameId(raw.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise StaleTeacherError(f"{path} records an unknown game") from error


def _uuids_of(column: pa.ChunkedArray) -> tuple[UUID, ...]:
    """A nocab_uuid column back as UUIDs (via fixed_width_bytes_of).

    Private helper - single caller is TeacherTable.open.
    Inputs: column. Output: tuple[UUID, ...]. Side effects: none.
    Exceptions: none.
    """
    uuid_bytes = fixed_width_bytes_of(column, UUID_WIDTH)
    return tuple(UUID(bytes=row.tobytes()) for row in uuid_bytes)


def _vectors_of(column: pa.ChunkedArray) -> np.ndarray:
    """A vector column back as a read-only float32 (rows, dim) view of
    its one values buffer.

    Private helper - single caller is TeacherTable.open.
    Inputs: column. Output: np.ndarray. Side effects: none.
    Exceptions: none.
    """
    if column.num_chunks != 1:
        raise StaleTeacherError(f"vector column has {column.num_chunks} chunks, not 1")
    chunk = column.chunk(0)
    dim = chunk.type.list_size
    values = chunk.flatten().to_numpy()
    return values.reshape(len(chunk), dim)


def stamp_tag_of(payload: object) -> str:
    """The first _STAMP_TAG_LENGTH hex digits of payload's JSON sha256:
    the one tagging rule for teacher and neighbours file names.

    Inputs: payload (JSON-serialisable). Output: str.
    Side effects: none. Exceptions: TypeError if payload is not JSON-ready.

    Example:
        >>> stamp_tag_of({"kind": "top_k", "k": 32})
        'a1b2c3d4e5f6'
    """
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:_STAMP_TAG_LENGTH]
