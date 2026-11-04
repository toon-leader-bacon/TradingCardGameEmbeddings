"""CardSnapshot: the reader of one game's level-2 card snapshot.

cards.arrow is the saved index -> uuid table every other level-2 file
indexes into: one row per card of the game, ordered by nocab_uuid
bytes, the row number being the card's card_index. nocab_uuid is the
permanent identity (binder, deck box, metrics); card_index is valid only
for one card snapshot, identified by its stamp. This file is the only
translation between them.

Columns: nocab_uuid fixed_size_binary(16), is_unknown bool (exactly one
row, the game's Unknown sentinel).
"""

from pathlib import Path
from uuid import UUID

import numpy as np
import pyarrow as pa

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.snapshots.arrow_files import (
    fixed_width_bytes_of,
    open_mapped_table,
)
from src.data_refinement.snapshots.columns import (
    IS_UNKNOWN_COLUMN,
    NOCAB_UUID_COLUMN,
    UUID_WIDTH,
)
from src.data_refinement.snapshots.stamps import (
    BUILD_COMMAND,
    CARD_SNAPSHOT_FORMAT_VERSION,
    SNAPSHOT_ROOT,
    CardSnapshotStamp,
    StaleSnapshotError,
)
from src.schema.game_id import GameId

CARD_SNAPSHOT_FILE_NAME = "cards.arrow"


class CardSnapshot:
    """One game's card snapshot, memory-mapped and checked against the live
    binder. One per game per process (through CardShelf).

    source_game: the game. stamp: what the file was built from.
    card_count: rows in the file. card_uuids: index -> UUID, built once at
    open. unknown_index: the sentinel's card_index.
    """

    def __init__(
        self,
        source_game: GameId,
        stamp: CardSnapshotStamp,
        card_uuids: list[UUID],
        unknown_index: int,
        table: pa.Table,
    ) -> None:
        """Hold an already-read snapshot; use open() to read one.

        Inputs: as the class docstring; table (kept so the mapped file
            stays alive).
        Output: none. Side effects: none. Exceptions: none.
        """
        self.source_game = source_game
        self.stamp = stamp
        self.card_uuids = card_uuids
        self.unknown_index = unknown_index
        self._table = table

    @property
    def card_count(self) -> int:
        """Rows in the snapshot. Inputs: none. Output: int.
        Side effects: none. Exceptions: none."""
        return len(self.card_uuids)

    @staticmethod
    def default_directory(source_game: GameId) -> Path:
        """Where source_game's card snapshot lives:
        data/derived/card_snapshots/<game>/.

        Inputs: source_game. Output: Path. Side effects: none.
        Exceptions: none.

        Example:
            >>> CardSnapshot.default_directory(GameId.MTG)
            PosixPath('data/derived/card_snapshots/mtg')
        """
        return SNAPSHOT_ROOT / "card_snapshots" / source_game.value

    @classmethod
    def open(
        cls, directory: Path, card_lookup: CardLookup, source_game: GameId
    ) -> "CardSnapshot":
        """Memory-map directory/cards.arrow and check it against the binder.

        Inputs:
            directory: the snapshot's directory (default_directory's).
            card_lookup: the live binder holding source_game.
            source_game: whose snapshot.
        Output: CardSnapshot.
        Side effects: maps the file; builds card_count UUIDs.
        Exceptions: StaleSnapshotError if the file is missing, its stamp is
            unreadable, its format_version is not current, or its
            card_binder_version differs from card_lookup.version_for(
            source_game); also if it does not hold exactly one sentinel.

        Example:
            >>> snapshot = CardSnapshot.open(
            ...     CardSnapshot.default_directory(GameId.MTG), binder, GameId.MTG
            ... )
        """
        # Map the file and read its stamp
        table = open_snapshot_table(directory / CARD_SNAPSHOT_FILE_NAME)
        stamp = CardSnapshotStamp.from_schema_metadata(table.schema.metadata)

        # Validate: the file was built from this binder, in this format
        _require_current_stamp(stamp, card_lookup.version_for(source_game))

        # Build the index -> uuid list once, and find the sentinel
        card_uuids = _card_uuids_of(table)
        unknown_index = _unknown_index_of(table)
        return cls(source_game, stamp, card_uuids, unknown_index, table)


def open_snapshot_table(path: Path) -> pa.Table:
    """open_mapped_table for a level-2 snapshot file, where a missing or
    malformed file means the snapshot must be (re)built.

    Shared by CardSnapshot.open and DeckSnapshot.open.
    Inputs: path. Output: pa.Table (mapped).
    Side effects: maps the file.
    Exceptions: StaleSnapshotError naming the build command.

    Example:
        >>> table = open_snapshot_table(directory / CARD_SNAPSHOT_FILE_NAME)
    """
    try:
        return open_mapped_table(path)
    except (FileNotFoundError, ValueError) as error:
        raise StaleSnapshotError(
            f"{error}; build the snapshot with {BUILD_COMMAND}"
        ) from error


def _require_current_stamp(stamp: CardSnapshotStamp, card_binder_version: str) -> None:
    """Raise unless stamp matches the live binder version and the current
    format.

    Private helper - single consumer is CardSnapshot.open.
    Inputs: stamp (the file's), card_binder_version (the live binder's).
    Output: none. Side effects: none.
    Exceptions: StaleSnapshotError naming the build command.
    """
    if stamp.format_version != CARD_SNAPSHOT_FORMAT_VERSION:
        raise StaleSnapshotError(
            f"card snapshot format {stamp.format_version} is not the current "
            f"{CARD_SNAPSHOT_FORMAT_VERSION}; rebuild with {BUILD_COMMAND}"
        )
    if stamp.card_binder_version != card_binder_version:
        raise StaleSnapshotError(
            "card snapshot was built from another card binder version "
            f"({stamp.card_binder_version[:12]}..., binder is now "
            f"{card_binder_version[:12]}...); rebuild with {BUILD_COMMAND}"
        )


def _card_uuids_of(table: pa.Table) -> list[UUID]:
    """The nocab_uuid column as UUIDs, in row order, read from the column's
    data buffer (no per-row pyarrow scalar).

    Private helper - single consumer is CardSnapshot.open.
    Inputs: table. Output: list[UUID]. Side effects: none.
    Exceptions: none.
    """
    uuid_bytes = fixed_width_bytes_of(table.column(NOCAB_UUID_COLUMN), UUID_WIDTH)
    return [UUID(bytes=row.tobytes()) for row in uuid_bytes]


def _unknown_index_of(table: pa.Table) -> int:
    """The row whose is_unknown is true.

    Private helper - single consumer is CardSnapshot.open.
    Inputs: table. Output: int.
    Side effects: none.
    Exceptions: StaleSnapshotError unless exactly one row is the sentinel.
    """
    is_unknown = table.column(IS_UNKNOWN_COLUMN).to_numpy()
    unknown_rows = np.flatnonzero(is_unknown)
    if len(unknown_rows) != 1:
        raise StaleSnapshotError(
            f"card snapshot holds {len(unknown_rows)} Unknown sentinel rows, "
            f"not 1; rebuild with {BUILD_COMMAND}"
        )
    return int(unknown_rows[0])
