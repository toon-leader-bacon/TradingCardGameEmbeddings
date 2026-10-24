"""SqliteFeatureStore: the durable disk tier of the feature cache.

One SQLite file holds any number of row sets, one per
FrozenStageFingerprint digest, so encoded cards survive across runs and
experiments. It keeps only durable puts, admits until its row set is full
and never evicts. The connection opens on first use, so building a model
creates no file.

Schema:
    row_sets(digest TEXT PRIMARY KEY, description TEXT, width INTEGER,
             dtype TEXT, created_at TEXT)
    features(digest TEXT, key BLOB, row BLOB,
             PRIMARY KEY (digest, key)) WITHOUT ROWID
A row is the raw bytes of one storage-dtype vector.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.feature_store import (
    StoreLookup,
    TierStats,
    require_offer_shape,
)
from src.encoder_model.feature_cache.frozen_stage_fingerprint import (
    FrozenStageFingerprint,
)

# Keys per IN (...) query, well under SQLite's bound-variable limit
_LOOKUP_BATCH = 500
# Milliseconds a writer waits for another process's write lock
_BUSY_TIMEOUT_MS = 30_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS row_sets (
    digest TEXT PRIMARY KEY, description TEXT NOT NULL, width INTEGER NOT NULL,
    dtype TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS features (
    digest TEXT NOT NULL, key BLOB NOT NULL, row BLOB NOT NULL,
    PRIMARY KEY (digest, key)
) WITHOUT ROWID;
"""


class SqliteFeatureStore:
    """FeatureStore[Tensor] in one SQLite file, scoped to one row set."""

    def __init__(
        self, path: Path, fingerprint: FrozenStageFingerprint, capacity_rows: int
    ) -> None:
        """Inputs: path (the database file; may not exist yet), fingerprint
            (names and describes this store's row set; width and storage
            dtype come from it), capacity_rows (>= 1, for this row set).
        Output: none (constructor). Side effects: none; no I/O until the
            first lookup or put_many.
        Exceptions: ValueError if capacity_rows < 1, or the fingerprint's
            text encoder has no source or no revision (durable rows must
            name the exact weights that made them).

        Example:
            >>> SqliteFeatureStore(Path("data/feature_cache/features.sqlite"), fingerprint, 10**6)
        """
        if capacity_rows < 1:
            raise ValueError(f"capacity_rows must be >= 1, got {capacity_rows}")
        _require_pinned_source(fingerprint)
        self._path = path
        self._fingerprint = fingerprint
        self._digest = fingerprint.digest()
        self._capacity_rows = capacity_rows
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        self._closed = False
        self._rows_used = 0
        self._hits = 0

    def lookup(
        self, keys: Sequence[CardTextKey], device: torch.device
    ) -> StoreLookup[torch.Tensor]:
        """The stored rows of keys, read in batches.

        Inputs: keys (distinct), device.
        Output: StoreLookup; hits (len(hit_positions), width) in the storage
            dtype on device.
        Side effects: opens the database on first use; counts hits.
        Exceptions: RuntimeError after close(); sqlite3.Error from the
            database.

        Example:
            >>> store.lookup([held_key, new_key], torch.device("cuda")).miss_positions
            [1]
        """
        with self._lock:
            connection = self._open()
            # Read every stored row, then split by position
            found = _select_rows(connection, self._digest, keys)
            hit_positions = [i for i, key in enumerate(keys) if key in found]
            miss_positions = [i for i, key in enumerate(keys) if key not in found]
            self._hits += len(hit_positions)
        if not hit_positions:
            return StoreLookup([], None, miss_positions)
        rows = [found[keys[i]] for i in hit_positions]
        result = StoreLookup(
            hit_positions, self._decode(rows).to(device), miss_positions
        )
        return result

    def put_many(
        self, keys: Sequence[CardTextKey], features: torch.Tensor, durable: bool
    ) -> int:
        """Store durable rows for keys this row set lacks, while it has room.

        Inputs: keys (distinct), features (len(keys), width) on any device,
            durable (a non-durable put is ignored).
        Output: int, rows newly written.
        Side effects: opens the database on first use; one transaction.
        Exceptions: ValueError for a length or width mismatch;
            RuntimeError after close() (for a durable, non-empty put);
            sqlite3.Error from the database.

        Example:
            >>> store.put_many([key_a], torch.randn(1, 768), durable=True)
            1
        """
        require_offer_shape(keys, features, self._fingerprint.width)
        if not durable or not keys:
            return 0
        with self._lock:
            connection = self._open()
            if self._rows_used >= self._capacity_rows:
                return 0
            payload = self._encode(features)
            # Skip held keys, then take only what the row set has room for
            held = _select_held_keys(connection, self._digest, keys)
            fresh = [i for i, key in enumerate(keys) if key not in held]
            fresh = fresh[: max(0, self._capacity_rows - self._rows_used)]
            written = _insert_rows(
                connection, self._digest, [(keys[i], payload[i]) for i in fresh]
            )
            self._rows_used += written
        return written

    def pin(self, keys: Sequence[CardTextKey]) -> None:
        """Nothing: the disk never evicts, so pins do not apply.
        Inputs: keys. Output: none. Side effects: none. Exceptions: none."""

    def unpin(self, keys: Sequence[CardTextKey]) -> None:
        """Nothing: the disk never evicts, so pins do not apply.
        Inputs: keys. Output: none. Side effects: none. Exceptions: none."""

    def tier_stats(self) -> tuple[TierStats, ...]:
        """The disk tier's totals (rows_used is 0 until the database is
        opened). Inputs: none. Output: one TierStats named "disk".
        Side effects: none. Exceptions: none."""
        with self._lock:
            return (
                TierStats(
                    "disk", self._hits, self._rows_used, self._capacity_rows, 0, 0
                ),
            )

    @property
    def storage_dtype(self) -> torch.dtype:
        """The fingerprint's storage dtype. Inputs: none. Output:
        torch.dtype. Side effects: none. Exceptions: none."""
        return self._fingerprint.storage_dtype.torch_dtype

    def close(self) -> None:
        """Close the connection, if it was opened. Inputs: none.
        Output: none. Side effects: closes the database connection and marks the
        store closed: later lookups and durable puts raise RuntimeError.
        Exceptions: none."""
        with self._lock:
            self._closed = True
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _open(self) -> sqlite3.Connection:
        """The connection, opened, configured and registered on first use.

        Private helper - consumers are lookup and put_many (lock held).
        Inputs: none. Output: sqlite3.Connection.
        Side effects: on first call: creates the parent directory and
            the file if absent, then _prepare; a failed preparation closes
            the connection again.
        Exceptions: RuntimeError after close(); ValueError if the row set
            is registered with another width or dtype; sqlite3.Error.
        """
        if self._closed:
            raise RuntimeError(f"the disk feature cache at {self._path} is closed")
        if self._connection is not None:
            return self._connection
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, check_same_thread=False)
        try:
            self._rows_used = _prepare(connection, self._digest, self._fingerprint)
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        return connection

    def _encode(self, features: torch.Tensor) -> list[bytes]:
        """One bytes value per row, in the storage dtype.

        Private helper - single consumer is put_many.
        Inputs: features (N, width). Output: list of N bytes.
        Side effects: copies features to the CPU. Exceptions: none.
        """
        stored = features.detach().to("cpu", self.storage_dtype).contiguous()
        raw = stored.view(torch.uint8).numpy()
        return [row.tobytes() for row in raw]

    def _decode(self, rows: list[bytes]) -> torch.Tensor:
        """The (len(rows), width) CPU tensor of stored rows, in order.

        Private helper - single consumer is lookup.
        Inputs: rows (each width * itemsize bytes). Output: Tensor in the
            storage dtype. Side effects: none. Exceptions: none.
        """
        buffer = bytearray(b"".join(rows))
        flat = torch.frombuffer(buffer, dtype=self.storage_dtype)
        return flat.reshape(len(rows), self._fingerprint.width)


def _select_rows(
    connection: sqlite3.Connection, digest: str, keys: Sequence[CardTextKey]
) -> dict[bytes, bytes]:
    """The stored rows of keys, by key.

    Private helper - single consumer is SqliteFeatureStore.lookup.
    Inputs: connection, digest (the row set), keys.
    Output: dict key bytes -> row bytes, for the keys stored.
    Side effects: none. Exceptions: sqlite3.Error.
    """
    query = "SELECT key, row FROM features WHERE digest = ? AND key IN ({marks})"
    return dict(_query_in_batches(connection, query, digest, keys))


def _select_held_keys(
    connection: sqlite3.Connection, digest: str, keys: Sequence[CardTextKey]
) -> set[bytes]:
    """The keys this row set already holds, without reading
    their rows.

    Private helper - single consumer is SqliteFeatureStore.put_many.
    Inputs: connection, digest (the row set), keys.
    Output: set of key bytes. Side effects: none.
    Exceptions: sqlite3.Error.
    """
    query = "SELECT key FROM features WHERE digest = ? AND key IN ({marks})"
    return {row[0] for row in _query_in_batches(connection, query, digest, keys)}


def _query_in_batches(
    connection: sqlite3.Connection,
    query: str,
    digest: str,
    keys: Sequence[CardTextKey],
) -> list[tuple]:
    """Run query (with a `{marks}` slot for an IN list) over keys,
    _LOOKUP_BATCH keys at a time, and gather every result row.

    Private helper - consumers are _select_rows and _select_held_keys.
    Inputs: connection, query, digest (the first parameter), keys.
    Output: list of result tuples. Side effects: none.
    Exceptions: sqlite3.Error.
    """
    result: list[tuple] = []
    for start in range(0, len(keys), _LOOKUP_BATCH):
        batch = keys[start : start + _LOOKUP_BATCH]
        marks = ",".join("?" * len(batch))
        result.extend(connection.execute(query.format(marks=marks), (digest, *batch)))
    return result


def _insert_rows(
    connection: sqlite3.Connection, digest: str, rows: list[tuple[bytes, bytes]]
) -> int:
    """Insert (key, row bytes) pairs in one transaction, with
    INSERT OR IGNORE: another process may have written the same key since
    put_many checked. Counts only rows actually inserted (the
    total_changes delta), so a row set's fill stays honest.

    Private helper - single consumer is SqliteFeatureStore.put_many.
    Inputs: connection, digest, rows. Output: int, rows inserted.
    Side effects: one committed transaction. Exceptions: sqlite3.Error.
    """
    if not rows:
        return 0
    before = connection.total_changes
    with connection:
        connection.executemany(
            "INSERT OR IGNORE INTO features VALUES (?, ?, ?)",
            [(digest, key, row) for key, row in rows],
        )
    return connection.total_changes - before


def _require_pinned_source(fingerprint: FrozenStageFingerprint) -> None:
    """Private helper - single consumer is SqliteFeatureStore.__init__.

    Inputs: fingerprint. Output: none. Side effects: none.
    Exceptions: ValueError if fingerprint.text_encoder is None or its
        revision is None.
    """
    source = fingerprint.text_encoder
    if source is None or source.revision is None:
        raise ValueError(
            "a disk feature cache needs a text encoder with a known revision "
            "(a Hugging Face checkpoint); a local path's weights can change "
            "under the same name"
        )


def _require_row_set_matches(
    connection: sqlite3.Connection, digest: str, fingerprint: FrozenStageFingerprint
) -> None:
    """A sanity check: the registered row set has the fingerprint's width
    and dtype (the digest covers both, so a mismatch means a corrupt file).

    Private helper - single consumer is SqliteFeatureStore._open.
    Inputs: connection, digest, fingerprint. Output: none.
    Side effects: none. Exceptions: ValueError on a mismatch.
    """
    width, dtype = connection.execute(
        "SELECT width, dtype FROM row_sets WHERE digest = ?", (digest,)
    ).fetchone()
    expected = (fingerprint.width, fingerprint.storage_dtype.value)
    if (width, dtype) != expected:
        raise ValueError(
            f"row set {digest[:12]} is registered as {(width, dtype)}, not {expected}"
        )


def _prepare(
    connection: sqlite3.Connection, digest: str, fingerprint: FrozenStageFingerprint
) -> int:
    """Configure a fresh connection, create the schema if absent, register
    the row set and count its rows.

    Private helper - single consumer is SqliteFeatureStore._open.
    Inputs: connection (just opened), digest, fingerprint.
    Output: int, the row set's stored rows.
    Side effects: sets WAL, synchronous=NORMAL and the busy timeout;
        creates tables; inserts the row set if new.
    Exceptions: ValueError if the row set is registered with another width
        or dtype; sqlite3.Error.
    """
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    with connection:
        connection.executescript(_SCHEMA)
        connection.execute(
            "INSERT OR IGNORE INTO row_sets VALUES (?, ?, ?, ?, ?)",
            (
                digest,
                fingerprint.describe(),
                fingerprint.width,
                fingerprint.storage_dtype.value,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    _require_row_set_matches(connection, digest, fingerprint)
    (rows_used,) = connection.execute(
        "SELECT COUNT(*) FROM features WHERE digest = ?", (digest,)
    ).fetchone()
    return int(rows_used)
