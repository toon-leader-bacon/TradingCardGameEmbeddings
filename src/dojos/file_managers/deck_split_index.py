"""DeckSplitIndex: one game's TRAIN/TEST/VALIDATION deck rows under one
HoldoutSpec, numbered by a deck snapshot's rows.

The split comes from holdout.deck_split_of over every deck of the
snapshot (about 10 s for MTG's 6.9M decks), cached on disk as one small
Arrow file per game and deck split key: a uint8 split code per deck row.
Per run (it depends on the holdout), so it lives with the dojos, not in
level 2; CardShelf holds one per (game, deck_split_key), shared by every
contrastive dojo of that game.

Because it stores deck ROW NUMBERS, it must not outlive the snapshot it
numbered: the file records the DeckSnapshotStamp it was built from, and
load_or_build rebuilds when that differs from the open snapshot's.
"""

import hashlib
from pathlib import Path
from uuid import UUID

import numpy as np
import pyarrow as pa

from src.data_refinement.snapshots.arrow_files import (
    open_mapped_table,
    write_table_atomically,
)
from src.data_refinement.snapshots.columns import UUID_WIDTH
from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.data_refinement.snapshots.stamps import DeckSnapshotStamp, StaleSnapshotError
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

# Hex digits of the deck split key's sha256 in a cache file's name
_SPLIT_TAG_LENGTH = 12

# The on-disk code of each split. Persisted, so explicit (never enum
# order); changing it means bumping _CACHE_FORMAT_VERSION
_SPLIT_CODES: dict[Split, int] = {Split.TRAIN: 0, Split.TEST: 1, Split.VALIDATION: 2}

# Recorded in each cache file; a file with another version is a miss
_CACHE_FORMAT_VERSION = 1

# The cache file: its name's suffix, its one column, its metadata keys
# (beside the deck snapshot stamp's own)
_CACHE_SUFFIX = ".deck_splits.arrow"
_SPLIT_CODE_COLUMN = "split_code"
_DECK_SPLIT_KEY_KEY = b"deck_split_key"
_CACHE_FORMAT_VERSION_KEY = b"deck_split_cache_format_version"


def deck_split_tag(deck_split_key: str) -> str:
    """A short, file-name-safe tag of one game's deck split.

    Inputs: deck_split_key (HoldoutSpec.deck_split_key's). Output: str
    (hex). Side effects: none. Exceptions: none.

    Example:
        >>> deck_split_tag(holdout.deck_split_key(GameId.MTG))
        '3f2b1c0a9d8e'
    """
    return hashlib.sha256(deck_split_key.encode()).hexdigest()[:_SPLIT_TAG_LENGTH]


class DeckSplitIndex:
    """One game's deck split over one deck snapshot's rows.

    deck_split_key: the holdout's key for the game. snapshot_stamp: the
    snapshot whose rows it numbers. split_codes: uint8 per deck row
    (_SPLIT_CODES), made read-only so rows_of always agrees with it.
    Each split's rows are computed once, at construction.
    """

    def __init__(
        self,
        deck_split_key: str,
        snapshot_stamp: DeckSnapshotStamp,
        split_codes: np.ndarray,
    ) -> None:
        """Hold an already-computed split; use load_or_build() to get one.

        Inputs: as the class docstring. Output: none. Side effects: none.
        Exceptions: none.
        """
        self.deck_split_key = deck_split_key
        self.snapshot_stamp = snapshot_stamp
        self.split_codes = split_codes
        self.split_codes.setflags(write=False)
        self._rows: dict[Split, np.ndarray] = _rows_by_split(split_codes)

    @classmethod
    def load_or_build(
        cls, snapshot: DeckSnapshot, holdout: HoldoutSpec, cache_directory: Path
    ) -> "DeckSplitIndex":
        """The cached index for (snapshot's game, deck_split_key) if it
        records snapshot.stamp; otherwise build and cache it.

        Inputs:
            snapshot: the game's open deck snapshot.
            holdout: the run's spec; holdout.deck_split_of assigns splits.
            cache_directory: where the cache file lives
                (<game>.<split tag>.deck_splits.arrow).
        Output: DeckSplitIndex.
        Side effects: may read the cache file; on a miss computes every
            deck's split and writes the cache file atomically.
        Exceptions: UnknownDeckSplitError if holdout has no deck split
            (before any I/O); OSError on a write failure.

        Example:
            >>> index = DeckSplitIndex.load_or_build(decks, holdout, cache_dir)
            >>> index.rows_of(Split.TRAIN)
        """
        # Validate inputs: an unknown deck split raises here, before any I/O
        deck_split_key = holdout.deck_split_key(snapshot.source_game)
        path = _cache_path(cache_directory, snapshot, deck_split_key)

        # Reuse the cache when it numbers this snapshot's rows
        cached = _read_cache(path, deck_split_key)
        if cached is not None and cached.snapshot_stamp == snapshot.stamp:
            return cached

        # Otherwise assign every deck and cache the result
        result = cls(
            deck_split_key, snapshot.stamp, _assign_split_codes(snapshot, holdout)
        )
        _write_cache(path, result)
        return result

    def rows_of(self, split: Split) -> np.ndarray:
        """The deck rows assigned to split, ascending (deck uuid order).

        Inputs: split. Output: np.ndarray of int64, read-only and shared
            between calls (callers copy before reordering).
        Side effects: none. Exceptions: none.

        Example:
            >>> len(index.rows_of(Split.TEST))
            690000
        """
        return self._rows[split]


def _rows_by_split(split_codes: np.ndarray) -> dict[Split, np.ndarray]:
    """Each split's rows (read-only int64 arrays), from the codes.

    Private helper - single consumer is DeckSplitIndex.__init__.
    Inputs: split_codes. Output: dict over every Split.
    Side effects: none. Exceptions: none.
    """
    result: dict[Split, np.ndarray] = {}
    for split, code in _SPLIT_CODES.items():
        rows = np.flatnonzero(split_codes == code).astype(np.int64)
        rows.setflags(write=False)
        result[split] = rows
    return result


def _cache_path(
    cache_directory: Path, snapshot: DeckSnapshot, deck_split_key: str
) -> Path:
    """<cache_directory>/<game>.<split tag>.deck_splits.arrow.

    Private helper - single consumer is load_or_build.
    Inputs: cache_directory, snapshot, deck_split_key. Output: Path.
    Side effects: none. Exceptions: none.
    """
    file_name = (
        f"{snapshot.source_game.value}.{deck_split_tag(deck_split_key)}"
        f"{_CACHE_SUFFIX}"
    )
    return cache_directory / file_name


def _read_cache(path: Path, deck_split_key: str) -> DeckSplitIndex | None:
    """The cached index at path, or None if the file is missing,
    unreadable (arrow_files' FileNotFoundError / ValueError, or a stamp
    StaleSnapshotError), or records another deck_split_key or
    _CACHE_FORMAT_VERSION.

    Private helper - single consumer is load_or_build.
    Inputs: path, deck_split_key. Output: DeckSplitIndex | None.
    Side effects: reads path. Exceptions: none (an unreadable file is a
        miss).
    """
    try:
        table = open_mapped_table(path)
        metadata = table.schema.metadata or {}
        snapshot_stamp = DeckSnapshotStamp.from_schema_metadata(metadata)
    except (FileNotFoundError, ValueError, StaleSnapshotError):
        return None

    # A cache of another split or format is a miss, never a reuse
    recorded_key = metadata.get(_DECK_SPLIT_KEY_KEY)
    recorded_format = metadata.get(_CACHE_FORMAT_VERSION_KEY)
    if (
        recorded_key != deck_split_key.encode()
        or recorded_format != str(_CACHE_FORMAT_VERSION).encode()
    ):
        return None

    split_codes = table.column(_SPLIT_CODE_COLUMN).to_numpy()
    return DeckSplitIndex(deck_split_key, snapshot_stamp, split_codes)


def _assign_split_codes(snapshot: DeckSnapshot, holdout: HoldoutSpec) -> np.ndarray:
    """holdout.deck_split_of for every deck row, as uint8 codes.

    Private helper - single consumer is load_or_build.
    Inputs: snapshot, holdout. Output: np.ndarray uint8 (deck_count,).
    Side effects: none.
    Exceptions: UnknownDeckSplitError as holdout.deck_split_of.
    """
    source_game = snapshot.source_game
    result = np.empty(snapshot.deck_count, dtype=np.uint8)

    # One bytes object for every uuid; a slice per deck, no NumPy scalar
    uuid_bytes = snapshot.deck_uuid_bytes.tobytes()
    for row in range(snapshot.deck_count):
        deck_uuid = UUID(bytes=uuid_bytes[row * UUID_WIDTH : (row + 1) * UUID_WIDTH])
        result[row] = _SPLIT_CODES[holdout.deck_split_of(deck_uuid, source_game)]
    return result


def _write_cache(path: Path, index: DeckSplitIndex) -> None:
    """Write index to path atomically, its deck_split_key, snapshot stamp
    and _CACHE_FORMAT_VERSION in the schema metadata.

    Private helper - single consumer is load_or_build.
    Inputs: path, index. Output: none.
    Side effects: writes path (and its directory).
    Exceptions: OSError on a write failure.
    """
    metadata = {
        **index.snapshot_stamp.to_schema_metadata(),
        _DECK_SPLIT_KEY_KEY: index.deck_split_key.encode(),
        _CACHE_FORMAT_VERSION_KEY: str(_CACHE_FORMAT_VERSION).encode(),
    }
    schema = pa.schema([(_SPLIT_CODE_COLUMN, pa.uint8())], metadata=metadata)
    table = pa.Table.from_arrays([pa.array(index.split_codes)], schema=schema)
    write_table_atomically(table, path)
