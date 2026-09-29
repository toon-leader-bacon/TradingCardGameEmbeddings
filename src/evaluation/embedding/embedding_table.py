"""EmbeddingTable: one encoder's card embeddings, stored once in SQLite so
any number of intrinsic analyses can read them.

One table = one encoder; comparing encoders means comparing tables built
from the same corpus. Same storage precedent as DeckBox (SQLite), but
unlike DeckBox every add() is committed before it returns: embed_corpus's
resume guarantee depends on it.

Schema:
    table_info(key TEXT PRIMARY KEY, value TEXT)  - "format" and "metadata"
    cards(nocab_uuid TEXT PRIMARY KEY, source_game TEXT, vector BLOB)
    vector is the row's float32 values as raw bytes (embedding_dim * 4).
"""

import sqlite3
from pathlib import Path
from typing import Sequence
from uuid import UUID

import numpy as np

from src.evaluation.card_row import CardRow
from src.evaluation.embedding.embedding_table_metadata import EmbeddingTableMetadata
from src.schema.game_id import GameId

# Written at create(), checked at open(): tells an EmbeddingTable apart
# from any other SQLite file (e.g. a DeckBox)
_FORMAT = "nocab.embedding_table.v1"

# SQLite's default limit on "?" parameters in one statement
_MAX_BOUND_PARAMETERS = 999


class NonFiniteEmbeddingError(ValueError):
    """Embedding vectors contain NaN or infinite values."""


def require_finite_vectors(vectors: np.ndarray) -> None:
    """The one finiteness check: guards the table's invariant in add(), and
    lets embed_corpus tell a skippable bad batch apart.

    Inputs: vectors (np.ndarray). Output: None. Side effects: none.
    Exceptions: NonFiniteEmbeddingError (a ValueError) if any value is NaN
        or infinite.

    Example:
        >>> require_finite_vectors(np.zeros((2, 4), dtype=np.float32))
    """
    finite = np.isfinite(vectors)
    if not finite.all():
        count = int(finite.size - finite.sum())
        raise NonFiniteEmbeddingError(f"{count} NaN or infinite values")


class EmbeddingTable:
    """An open embedding table. Build with create() or open(), never the
    constructor; use as a context manager (or call close()).

    Inputs (constructor, private): an open sqlite3.Connection and the
        table's parsed metadata.
    """

    def __init__(
        self, connection: sqlite3.Connection, metadata: EmbeddingTableMetadata
    ) -> None:
        self._connection = connection
        self._metadata = metadata

    @classmethod
    def create(cls, path: Path, metadata: EmbeddingTableMetadata) -> "EmbeddingTable":
        """Create a new, empty table file.

        Inputs: path (Path) of the new .db file; metadata.
        Output: the open EmbeddingTable.
        Side effects: creates path's parent directories and the file,
            writing the schema, the format marker and the metadata
            (committed); on a write failure the file is removed again.
        Exceptions: FileExistsError if path exists; sqlite3.Error on a
            write failure.

        Example:
            >>> with EmbeddingTable.create(Path("out/embeddings/a.db"), meta) as table:
            ...     embed_corpus(model, corpus, table, batch_size=64)
        """
        # Refuse to touch an existing file, even an empty one
        if path.exists():
            raise FileExistsError(f"{path} already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path))
        try:
            # sqlite3 runs the CREATEs outside a transaction; the cleanup
            # below, not the transaction, is what makes creation all-or-nothing
            with connection:
                _write_schema(connection)
                _write_table_info(connection, metadata)
        except BaseException:
            # Leave nothing behind, so the same path can be retried
            connection.close()
            path.unlink(missing_ok=True)
            raise
        return cls(connection, metadata)

    @classmethod
    def open(cls, path: Path) -> "EmbeddingTable":
        """Open an existing table file.

        Inputs: path (Path).
        Output: the open EmbeddingTable, metadata parsed.
        Side effects: opens a connection (read and write).
        Exceptions: FileNotFoundError if path is absent; ValueError if the
            file is not an EmbeddingTable (not SQLite, no format marker, a
            different format, or unparseable metadata).

        Example:
            >>> with EmbeddingTable.open(Path("out/embeddings/a.db")) as table:
            ...     rows = table.rows()
        """
        # sqlite3.connect would silently create a missing file
        if not path.exists():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(str(path))
        try:
            metadata = _read_table_info(connection, path)
        except BaseException:
            connection.close()
            raise
        return cls(connection, metadata)

    @property
    def metadata(self) -> EmbeddingTableMetadata:
        """The metadata recorded at create(). Inputs: none. Output:
        EmbeddingTableMetadata. Side effects: none. Exceptions: none."""
        return self._metadata

    def close(self) -> None:
        """Close the connection; safe to call twice. Afterwards every other
        method raises sqlite3.ProgrammingError. Inputs: none. Output: None.
        Side effects: closes the file. Exceptions: none."""
        self._connection.close()

    def __enter__(self) -> "EmbeddingTable":
        """Inputs: none. Output: self. Side effects: none. Exceptions:
        none."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the table; never suppresses the body's exception.
        Inputs: exception info (ignored). Output: None. Side effects:
        close(). Exceptions: none."""
        self.close()

    def has_card(self, nocab_uuid: UUID) -> bool:
        """Whether a vector for nocab_uuid is stored.

        Inputs: nocab_uuid (UUID). Output: bool. Side effects: none.
        Exceptions: sqlite3.Error on a read failure.

        Example:
            >>> table.has_card(card.nocab_uuid)
            True
        """
        found = self._connection.execute(
            "SELECT 1 FROM cards WHERE nocab_uuid = ?", (str(nocab_uuid),)
        ).fetchone()
        return found is not None

    def add(self, rows: Sequence[CardRow], vectors: np.ndarray) -> None:
        """Store one vector per row, committed before returning.

        Inputs: rows (Sequence[CardRow]); vectors (np.ndarray) of shape
            (len(rows), embedding_dim), cast to float32 for storage.
        Output: None.
        Side effects: inserts rows whose nocab_uuid is not yet stored; an
            already-present nocab_uuid (earlier call or earlier in rows) is
            left unchanged. One transaction: all or nothing.
        Exceptions: ValueError (nothing written) if vectors' shape is not
            (len(rows), embedding_dim), any row's source_game is not one
            of metadata.games, or any value is NaN or infinite after the
            float32 cast (NonFiniteEmbeddingError; a stored table is always
            finite); sqlite3.Error on a write failure (rolled
            back).

        Example:
            >>> table.add([CardRow(card.nocab_uuid, card.source_game)], vectors)
        """
        # Validate everything before writing anything
        self._require_vector_shape(len(rows), vectors)
        self._require_known_games(rows)
        # Overflow to inf is expected here and rejected on the next line, so
        # numpy's overflow warning would only be noise
        with np.errstate(over="ignore"):
            stored = np.asarray(vectors, dtype=np.float32)
        # After the cast: a finite float64 beyond float32's range becomes inf
        require_finite_vectors(stored)
        # Insert every row in one transaction, keeping existing rows as-is
        with self._connection:
            self._connection.executemany(
                "INSERT OR IGNORE INTO cards (nocab_uuid, source_game, vector) "
                "VALUES (?, ?, ?)",
                [
                    (str(row.nocab_uuid), row.source_game.value, vector.tobytes())
                    for row, vector in zip(rows, stored)
                ],
            )

    def rows(self) -> list[CardRow]:
        """Every stored card, sorted by nocab_uuid.

        The order depends only on the table's contents, never on insertion
        order, so seeded sampling picks the same cards from two tables
        built over the same corpus.

        Inputs: none. Output: list[CardRow]. Side effects: none.
        Exceptions: sqlite3.Error on a read failure.

        (ORDER BY the TEXT column is UUID order: the canonical lowercase
        form has fixed-width hex fields.)

        Example:
            >>> [row.source_game for row in table.rows()][:2]
            [<GameId.MTG: 'mtg'>, <GameId.GWENT: 'gwent'>]
        """
        cursor = self._connection.execute(
            "SELECT nocab_uuid, source_game FROM cards ORDER BY nocab_uuid"
        )
        return [CardRow(UUID(uuid_text), GameId(game)) for uuid_text, game in cursor]

    def vectors_for(self, rows: Sequence[CardRow]) -> np.ndarray:
        """The stored vectors of rows, in rows' order.

        Inputs: rows (Sequence[CardRow]), may be empty or repeat a card.
        Output: float32 np.ndarray (len(rows), embedding_dim).
        Side effects: none.
        Exceptions: KeyError naming the first nocab_uuid not stored;
            sqlite3.Error on a read failure.

        Example:
            >>> table.vectors_for(table.rows()[:10]).shape
            (10, 256)
        """
        result = np.empty((len(rows), self._metadata.embedding_dim), dtype=np.float32)
        # Fetch each distinct card once, in chunks under SQLite's
        # bound-parameter limit
        stored = self._vectors_by_uuid({row.nocab_uuid for row in rows})
        # Place each vector at its row's position
        for index, row in enumerate(rows):
            if row.nocab_uuid not in stored:
                raise KeyError(row.nocab_uuid)
            result[index] = stored[row.nocab_uuid]
        return result

    def _require_vector_shape(self, row_count: int, vectors: np.ndarray) -> None:
        """Inputs: row_count, vectors. Output: None. Side effects: none.
        Exceptions: ValueError naming both shapes unless vectors.shape ==
        (row_count, embedding_dim)."""
        expected = (row_count, self._metadata.embedding_dim)
        if vectors.shape != expected:
            raise ValueError(f"vectors have shape {vectors.shape}, expected {expected}")

    def _require_known_games(self, rows: Sequence[CardRow]) -> None:
        """Inputs: rows. Output: None. Side effects: none. Exceptions:
        ValueError naming the unknown games if any row's source_game is not
        one of metadata.games."""
        unknown = {row.source_game for row in rows} - self._metadata.games
        if unknown:
            names = sorted(game.value for game in unknown)
            raise ValueError(f"table was not created for games {names}")

    def _vectors_by_uuid(self, uuids: set[UUID]) -> dict[UUID, np.ndarray]:
        """Inputs: uuids (set[UUID]). Output: dict of the stored float32
        vectors of whichever uuids are present (absent ones are simply
        missing), queried in chunks of at most _MAX_BOUND_PARAMETERS.
        Side effects: none. Exceptions: sqlite3.Error on a read failure."""
        result: dict[UUID, np.ndarray] = {}
        texts = [str(nocab_uuid) for nocab_uuid in uuids]
        for start in range(0, len(texts), _MAX_BOUND_PARAMETERS):
            chunk = texts[start : start + _MAX_BOUND_PARAMETERS]
            # Only "?" placeholders are formatted in; values stay bound
            placeholders = ",".join("?" * len(chunk))
            cursor = self._connection.execute(
                f"SELECT nocab_uuid, vector FROM cards WHERE nocab_uuid IN ({placeholders})",
                chunk,
            )
            for uuid_text, blob in cursor:
                result[UUID(uuid_text)] = np.frombuffer(blob, dtype=np.float32)
        return result


def _write_schema(connection: sqlite3.Connection) -> None:
    """Inputs: an open connection. Output: None. Side effects: CREATEs
    the table_info and cards tables (see module docstring); does not
    commit (the caller's transaction). Exceptions: sqlite3.Error."""
    connection.execute(
        "CREATE TABLE table_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE cards (nocab_uuid TEXT PRIMARY KEY, "
        "source_game TEXT NOT NULL, vector BLOB NOT NULL)"
    )


def _write_table_info(
    connection: sqlite3.Connection, metadata: EmbeddingTableMetadata
) -> None:
    """Inputs: an open connection, metadata. Output: None. Side effects:
    inserts the "format" and "metadata" rows; does not commit (the
    caller's transaction). Exceptions: sqlite3.Error."""
    connection.executemany(
        "INSERT INTO table_info (key, value) VALUES (?, ?)",
        [("format", _FORMAT), ("metadata", metadata.to_json())],
    )


def _read_table_info(
    connection: sqlite3.Connection, path: Path
) -> EmbeddingTableMetadata:
    """Inputs: an open connection, its path (for messages). Output: the
    parsed EmbeddingTableMetadata. Side effects: reads table_info.
    Exceptions: ValueError naming path if the file is not SQLite
    (sqlite3.DatabaseError), has no table_info, has another format, or its
    metadata does not parse."""
    try:
        found = dict(connection.execute("SELECT key, value FROM table_info"))
    except sqlite3.DatabaseError as error:
        raise ValueError(f"{path} is not an EmbeddingTable") from error
    if found.get("format") != _FORMAT or "metadata" not in found:
        raise ValueError(f"{path} is not an EmbeddingTable ({_FORMAT})")
    try:
        return EmbeddingTableMetadata.from_json(found["metadata"])
    except ValueError as error:
        raise ValueError(f"{path} has unreadable metadata: {error}") from error
