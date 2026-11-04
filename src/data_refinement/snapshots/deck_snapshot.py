"""DeckSnapshot: the reader of one game's level-2 deck snapshot.

decks.arrow holds every deck of one game's deck box in CSR form, one row
per deck, ordered by deck nocab_uuid bytes:

- nocab_uuid fixed_size_binary(16)
- cards list<struct<card_index: uint16, count: uint8>>: one entry per
  distinct card, ascending by card_index (a row of the card snapshot the
  stamp names); counts clipped to 255 at build.

Nothing else about a deck is stored: training needs only the cards.

The reader exposes the buffers as NumPy views and never returns pyarrow
scalars: per-element pyarrow access in a loop is the one way to lose the
speed this file exists for.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa

from src.data_refinement.deck_box.deck_lookup import DeckLookup
from src.data_refinement.snapshots.arrow_files import fixed_width_bytes_of
from src.data_refinement.snapshots.card_snapshot import (
    CardSnapshot,
    open_snapshot_table,
)
from src.data_refinement.snapshots.columns import (
    CARD_INDEX_FIELD,
    CARDS_COLUMN,
    CARDS_TYPE,
    COUNT_FIELD,
    NOCAB_UUID_COLUMN,
    UUID_WIDTH,
)
from src.data_refinement.snapshots.stamps import (
    BUILD_COMMAND,
    DECK_SNAPSHOT_FORMAT_VERSION,
    SNAPSHOT_ROOT,
    DeckSnapshotStamp,
    StaleSnapshotError,
)
from src.schema.game_id import GameId

DECK_SNAPSHOT_FILE_NAME = "decks.arrow"


class DeckSnapshot:
    """One game's deck snapshot, memory-mapped and checked against the
    live deck box and the open card snapshot. One per game per process
    (through CardShelf), shared by every contrastive dojo of that game.

    stamp: what the file was built from. card_snapshot: the card snapshot
    its card_indices point into. deck_uuid_bytes: uint8 (deck_count, 16).
    deck_offsets: int32 (deck_count + 1,); deck d's cards are
    card_indices[deck_offsets[d]:deck_offsets[d + 1]], with card_counts
    aligned by construction (one struct column). All arrays are read-only
    views into the mapped file.
    """

    def __init__(
        self,
        stamp: DeckSnapshotStamp,
        card_snapshot: CardSnapshot,
        deck_uuid_bytes: np.ndarray,
        deck_offsets: np.ndarray,
        card_indices: np.ndarray,
        card_counts: np.ndarray,
        table: pa.Table,
    ) -> None:
        """Hold an already-read snapshot; use open() to read one.

        Inputs: as the class docstring; table (kept so the mapped file
            stays alive).
        Output: none. Side effects: none. Exceptions: none.
        """
        self.stamp = stamp
        self.card_snapshot = card_snapshot
        self.deck_uuid_bytes = deck_uuid_bytes
        self.deck_offsets = deck_offsets
        self.card_indices = card_indices
        self.card_counts = card_counts
        self._table = table

    @property
    def source_game(self) -> GameId:
        """The card snapshot's game. Inputs: none. Output: GameId.
        Side effects: none. Exceptions: none."""
        return self.card_snapshot.source_game

    @property
    def deck_count(self) -> int:
        """Decks in the snapshot. Inputs: none. Output: int.
        Side effects: none. Exceptions: none."""
        return len(self.deck_uuid_bytes)

    @staticmethod
    def default_directory(source_game: GameId) -> Path:
        """Where source_game's deck snapshot lives:
        data/derived/deck_snapshots/<game>/.

        Inputs: source_game. Output: Path. Side effects: none.
        Exceptions: none.

        Example:
            >>> DeckSnapshot.default_directory(GameId.MTG)
            PosixPath('data/derived/deck_snapshots/mtg')
        """
        return SNAPSHOT_ROOT / "deck_snapshots" / source_game.value

    @classmethod
    def open(
        cls, directory: Path, card_snapshot: CardSnapshot, deck_box: DeckLookup
    ) -> "DeckSnapshot":
        """Memory-map directory/decks.arrow and check it against level 1.

        Inputs:
            directory: the snapshot's directory (default_directory's).
            card_snapshot: the game's open card snapshot.
            deck_box: the live deck box the snapshot was built from.
        Output: DeckSnapshot.
        Side effects: maps the file; reads deck_box.version_for (a COUNT
            over the game's decks: about 48 s cold on MTG).
        Exceptions: StaleSnapshotError if the file is missing, its stamp is
            unreadable or of another format_version, its card_snapshot
            stamp differs from card_snapshot.stamp, or its
            deck_box_version differs from deck_box.version_for(game)
            (None, an unstamped box, never matches).

        Example:
            >>> decks = DeckSnapshot.open(
            ...     DeckSnapshot.default_directory(GameId.MTG), cards, box
            ... )
        """
        # Map the file and read its stamp
        table = open_snapshot_table(directory / DECK_SNAPSHOT_FILE_NAME)
        stamp = DeckSnapshotStamp.from_schema_metadata(table.schema.metadata)

        # Validate: built from this card snapshot and this deck box
        _require_current_stamp(
            stamp,
            card_snapshot,
            deck_box.version_for(card_snapshot.source_game),
        )

        # Wrap the columns' buffers as NumPy views
        deck_uuid_bytes = _deck_uuid_bytes_of(table)
        deck_offsets, card_indices, card_counts = _card_arrays_of(table)
        return cls(
            stamp,
            card_snapshot,
            deck_uuid_bytes,
            deck_offsets,
            card_indices,
            card_counts,
            table,
        )


def _require_current_stamp(
    stamp: DeckSnapshotStamp,
    card_snapshot: CardSnapshot,
    deck_box_version: str | None,
) -> None:
    """Raise unless stamp names card_snapshot's stamp, deck_box_version and
    the current format.

    Private helper - single consumer is DeckSnapshot.open.
    Inputs: stamp (the file's), card_snapshot (open), deck_box_version
        (the live box's; None never matches).
    Output: none. Side effects: none.
    Exceptions: StaleSnapshotError naming the build command.
    """
    if stamp.format_version != DECK_SNAPSHOT_FORMAT_VERSION:
        raise StaleSnapshotError(
            f"deck snapshot format {stamp.format_version} is not the current "
            f"{DECK_SNAPSHOT_FORMAT_VERSION}; rebuild with {BUILD_COMMAND}"
        )
    if stamp.card_snapshot != card_snapshot.stamp:
        raise StaleSnapshotError(
            f"{card_snapshot.source_game.value} deck snapshot indexes into another "
            f"card snapshot; rebuild with {BUILD_COMMAND}"
        )
    if stamp.deck_box_version != deck_box_version:
        raise StaleSnapshotError(
            f"{card_snapshot.source_game.value} deck snapshot was built from "
            f"another deck box version ({stamp.deck_box_version[:12]}..., box "
            f"is now {deck_box_version}); rebuild with {BUILD_COMMAND}"
        )


def _deck_uuid_bytes_of(table: pa.Table) -> np.ndarray:
    """The nocab_uuid column's data buffer as uint8 (deck_count, 16).

    Private helper - single consumer is DeckSnapshot.open.
    Inputs: table. Output: np.ndarray (read-only view).
    Side effects: none. Exceptions: none.
    """
    return fixed_width_bytes_of(table.column(NOCAB_UUID_COLUMN), UUID_WIDTH)


def _card_arrays_of(table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cards column as (offsets int32, card_index uint16, count
    uint8) views. Written as one unsliced batch, so the offsets index the
    struct children directly.

    Private helper - single consumer is DeckSnapshot.open.
    Inputs: table. Output: three read-only np.ndarray views.
    Side effects: none.
    Exceptions: ValueError if the column has more than one chunk.
    """
    column = table.column(CARDS_COLUMN)
    if column.num_chunks > 1:
        raise ValueError(f"expected one chunk, found {column.num_chunks}")
    if column.num_chunks == 0:
        cards = pa.array([], type=CARDS_TYPE)
    else:
        cards = column.chunk(0)

    # The offsets index into the struct child; zero-copy views of each part
    offsets = cards.offsets.to_numpy(zero_copy_only=True)
    entries = cards.values
    card_indices = entries.field(CARD_INDEX_FIELD).to_numpy(zero_copy_only=True)
    card_counts = entries.field(COUNT_FIELD).to_numpy(zero_copy_only=True)
    return offsets, card_indices, card_counts
