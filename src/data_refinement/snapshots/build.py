"""Compiling level 1 into the level-2 snapshots: build_card_snapshot
(binder -> cards.arrow) and build_deck_snapshot (deck box + card
snapshot -> decks.arrow). Each writes one file, stamped, replacing any
file already there.
"""

import logging
from array import array
from pathlib import Path
from uuid import UUID

import numpy as np
import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_lookup import DeckCardCounts, DeckLookup
from src.data_refinement.snapshots.arrow_files import write_table_atomically
from src.data_refinement.snapshots.card_snapshot import (
    CARD_SNAPSHOT_FILE_NAME,
    CardSnapshot,
)
from src.data_refinement.snapshots.columns import (
    CARDS_COLUMN,
    CARDS_TYPE,
    IS_UNKNOWN_COLUMN,
    NOCAB_UUID_COLUMN,
    UUID_WIDTH,
)
from src.data_refinement.snapshots.deck_snapshot import DECK_SNAPSHOT_FILE_NAME
from src.data_refinement.snapshots.stamps import (
    CARD_SNAPSHOT_FORMAT_VERSION,
    DECK_SNAPSHOT_FORMAT_VERSION,
    CardSnapshotStamp,
    DeckSnapshotStamp,
    StaleDeckBoxError,
)
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

# Card snapshot rows uint16 indices can number (exclusive cap: indices
# 0..65,535); the fix past it is uint32
_MAX_CARD_ROWS = int(np.iinfo(np.uint16).max) + 1

# Largest count uint8 holds (inclusive); larger counts clip to it
_MAX_COPY_COUNT = int(np.iinfo(np.uint8).max)

# The list column's int32 offsets cap a game's total deck entries
_MAX_DECK_ENTRIES = int(np.iinfo(np.int32).max)


def build_card_snapshot(
    card_lookup: CardLookup, source_game: GameId, directory: Path
) -> CardSnapshotStamp:
    """Write directory/cards.arrow for source_game, stamped; return the stamp.

    One row per card of the game plus the game's Unknown sentinel (added
    if the binder lacks it, so every deck card has an index to fall back
    to), ordered by nocab_uuid bytes.

    Inputs:
        card_lookup: the binder holding source_game.
        source_game: whose cards.
        directory: where cards.arrow goes (created if missing).
    Output: CardSnapshotStamp written into the file.
    Side effects: writes directory/cards.arrow atomically (replacing it).
    Exceptions: ValueError if the game has more cards than uint16 holds;
        OSError on a write failure.

    Example:
        >>> stamp = build_card_snapshot(
        ...     binder, GameId.MTG, CardSnapshot.default_directory(GameId.MTG)
        ... )
    """
    result = CardSnapshotStamp(
        card_binder_version=card_lookup.version_for(source_game),
        format_version=CARD_SNAPSHOT_FORMAT_VERSION,
    )

    # Every card of the game plus the sentinel, in uuid byte order
    card_uuids = _sorted_card_uuids(card_lookup, source_game)

    # Validate: indices must fit uint16
    if len(card_uuids) > _MAX_CARD_ROWS:
        raise ValueError(
            f"{source_game.value} has {len(card_uuids)} cards; uint16 card "
            f"indices hold {_MAX_CARD_ROWS} (widen card_index to uint32)"
        )

    # Write the table, stamped
    table = _card_table(card_uuids, source_game, result)
    write_table_atomically(table, directory / CARD_SNAPSHOT_FILE_NAME)
    return result


def build_deck_snapshot(
    deck_box: DeckLookup, card_snapshot: CardSnapshot, directory: Path
) -> DeckSnapshotStamp:
    """Write directory/decks.arrow for card_snapshot.source_game, stamped;
    return the stamp.

    Reads the deck box once, in its storage order (one sequential pass),
    then orders the decks by uuid bytes in memory. A count above 255 is
    clipped and a deck card missing from the card snapshot maps to the
    sentinel's index; each is tallied and logged once per build (a
    warning), never raised.

    Inputs:
        deck_box: the game's deck box.
        card_snapshot: the game's open card snapshot (indices point into it).
        directory: where decks.arrow goes (created if missing).
    Output: DeckSnapshotStamp written into the file.
    Side effects: reads every deck of the game; writes
        directory/decks.arrow atomically (replacing it); logs warnings.
    Exceptions: StaleDeckBoxError if deck_box.card_binder_version_for(game)
        is None or differs from card_snapshot.stamp.card_binder_version
        (when it is present, deck_box.version_for is never None);
        ValueError if the game's deck entries overflow int32 offsets, or
        if a deck is read twice; OSError on a write failure.

    Example:
        >>> stamp = build_deck_snapshot(box, cards, DeckSnapshot.default_directory(GameId.MTG))
    """
    source_game = card_snapshot.source_game

    # Validate: the box was minted against the binder the indices came from
    deck_box_version = _require_box_minted_against(deck_box, card_snapshot)
    result = DeckSnapshotStamp(
        card_snapshot=card_snapshot.stamp,
        deck_box_version=deck_box_version,
        format_version=DECK_SNAPSHOT_FORMAT_VERSION,
    )

    # Append every deck, in the box's order, as index/count entries
    columns = _DeckColumns(card_snapshot)
    for deck in deck_box.deck_card_counts(source_game):
        columns.append(deck)
    columns.log_irregular_entries(source_game)

    # Write the table, stamped
    write_table_atomically(
        columns.to_table(result), directory / DECK_SNAPSHOT_FILE_NAME
    )
    return result


class _DeckColumns:
    """The deck snapshot's columns as they are appended, one deck at a
    time, in growable typed buffers (never one Python object per entry
    kept), plus the tallies of clipped counts and unknown cards.

    Private helper - single consumer is build_deck_snapshot.
    """

    def __init__(self, card_snapshot: CardSnapshot) -> None:
        """Start empty, mapping uuids through card_snapshot.

        Inputs: card_snapshot. Output: none. Side effects: builds a
            UUID -> index dict over the card snapshot.
        Exceptions: none.
        """
        self._index_of = {
            card_uuid: index for index, card_uuid in enumerate(card_snapshot.card_uuids)
        }
        self._unknown_index = card_snapshot.unknown_index

        # The columns: 16 bytes per deck; int32 end offsets after a leading
        # 0; one uint16 index and one uint8 count per entry
        self._deck_uuid_bytes = bytearray()
        self._deck_offsets = array("i", [0])
        self._card_indices = array("H")
        self._card_counts = array("B")

        # Tallies for log_irregular_entries
        self._clipped_entries = 0
        self._largest_count = 0
        self._unknown_cards = 0

    def append(self, deck: DeckCardCounts) -> None:
        """Append one deck: its uuid bytes, then its cards as entries
        ascending by card_index (two unknown cards merge into one sentinel
        entry), counts clipped to 255; then its end offset.

        Inputs: deck (in any order; to_table sorts). Output: none.
        Side effects: grows the buffers and tallies.
        Exceptions: ValueError if total entries overflow int32 offsets.
        """
        self._deck_uuid_bytes += deck.nocab_uuid.bytes

        # Each distinct index once, ascending, its count clipped
        for index, count in sorted(self._merged_entries(deck).items()):
            if count > _MAX_COPY_COUNT:
                self._clipped_entries += 1
                self._largest_count = max(self._largest_count, count)
                count = _MAX_COPY_COUNT
            self._card_indices.append(index)
            self._card_counts.append(count)

        # Close the deck with its end offset
        if len(self._card_indices) > _MAX_DECK_ENTRIES:
            raise ValueError(
                f"over {_MAX_DECK_ENTRIES} deck entries overflow int32 offsets "
                "(widen the cards column to large_list)"
            )
        self._deck_offsets.append(len(self._card_indices))

    def log_irregular_entries(self, source_game: GameId) -> None:
        """Log one warning each (if any) for clipped counts (how many, and
        the largest count seen) and cards missing from the card snapshot.

        Inputs: source_game (named in the message). Output: none.
        Side effects: logs. Exceptions: none.
        """
        if self._clipped_entries:
            _logger.warning(
                "%s: clipped %d deck entries to a count of %d (largest count "
                "%d); the deck box keeps the true counts",
                source_game.value,
                self._clipped_entries,
                _MAX_COPY_COUNT,
                self._largest_count,
            )
        if self._unknown_cards:
            _logger.warning(
                "%s: %d deck cards are not in the card snapshot; mapped to the "
                "Unknown sentinel",
                source_game.value,
                self._unknown_cards,
            )

    def to_table(self, stamp: DeckSnapshotStamp) -> pa.Table:
        """The columns as one Arrow table, decks ordered by uuid bytes,
        stamp in its schema metadata.

        Inputs: stamp. Output: pa.Table (nocab_uuid, cards).
        Side effects: none.
        Exceptions: ValueError if a deck uuid was appended twice.
        """
        uuid_bytes = np.frombuffer(self._deck_uuid_bytes, dtype=np.uint8).reshape(
            -1, UUID_WIDTH
        )
        offsets = np.frombuffer(self._deck_offsets, dtype=np.int32)

        # Order the decks by uuid bytes, refusing a deck read twice
        order = _uuid_byte_order(uuid_bytes)
        sorted_uuid_bytes = uuid_bytes[order]
        _require_distinct_sorted(sorted_uuid_bytes)

        # Move each deck's entries with it
        sorted_offsets, entry_order = _reordered_entries(offsets, order)
        card_indices = np.frombuffer(self._card_indices, dtype=np.uint16)[entry_order]
        card_counts = np.frombuffer(self._card_counts, dtype=np.uint8)[entry_order]

        deck_uuids = pa.FixedSizeBinaryArray.from_buffers(
            pa.binary(UUID_WIDTH),
            len(order),
            [None, pa.py_buffer(sorted_uuid_bytes.tobytes())],
        )
        entries = pa.StructArray.from_arrays(
            [pa.array(card_indices), pa.array(card_counts)],
            fields=list(CARDS_TYPE.value_type),
        )
        cards = pa.ListArray.from_arrays(
            pa.array(sorted_offsets), entries, type=CARDS_TYPE
        )
        schema = pa.schema(
            [(NOCAB_UUID_COLUMN, pa.binary(UUID_WIDTH)), (CARDS_COLUMN, CARDS_TYPE)],
            metadata=stamp.to_schema_metadata(),
        )
        return pa.Table.from_arrays([deck_uuids, cards], schema=schema)

    def _merged_entries(self, deck: DeckCardCounts) -> dict[int, int]:
        """deck's counts keyed by card index; cards missing from the card
        snapshot are tallied and summed into the sentinel's index.

        Private helper - single consumer is append().
        Inputs: deck. Output: dict[int, int] (index -> total count).
        Side effects: tallies unknown cards. Exceptions: none.
        """
        result: dict[int, int] = {}
        for card_uuid, count in deck.card_counts:
            index = self._index_of.get(card_uuid)
            if index is None:
                self._unknown_cards += 1
                index = self._unknown_index
            result[index] = result.get(index, 0) + count
        return result


def _uuid_byte_order(uuid_bytes: np.ndarray) -> np.ndarray:
    """The permutation that sorts 16-byte rows by their bytes.

    Private helper - single consumer is _DeckColumns.to_table.
    Inputs: uuid_bytes (uint8, (n, 16)). Output: np.ndarray int64 (n,).
    Side effects: none. Exceptions: none.
    """
    # Two big-endian uint64 halves compare as the bytes do
    halves = np.ascontiguousarray(uuid_bytes).view(">u8")
    return np.lexsort((halves[:, 1], halves[:, 0]))


def _require_distinct_sorted(sorted_uuid_bytes: np.ndarray) -> None:
    """Raise if two adjacent sorted uuids are equal.

    Private helper - single consumer is _DeckColumns.to_table.
    Inputs: sorted_uuid_bytes (uint8, (n, 16)). Output: none.
    Side effects: none.
    Exceptions: ValueError naming the repeated deck: its card rows were
        not read as one run (DeckLookup.deck_card_counts's contract).
    """
    repeats = np.flatnonzero(
        np.all(sorted_uuid_bytes[1:] == sorted_uuid_bytes[:-1], axis=1)
    )
    if len(repeats):
        repeated = UUID(bytes=sorted_uuid_bytes[repeats[0]].tobytes())
        raise ValueError(
            f"deck {repeated} was read twice; the deck box must yield each deck once"
        )


def _reordered_entries(
    offsets: np.ndarray, order: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CSR offsets for the decks taken in order, and the gather index
    that moves every entry with its deck.

    Private helper - single consumer is _DeckColumns.to_table.
    Inputs: offsets (int32, (n + 1,)), order (a permutation of n decks).
    Output: (new offsets int32 (n + 1,), entry gather index int64).
    Side effects: none. Exceptions: none.
    """
    lengths = np.diff(offsets).astype(np.int64)[order]
    new_offsets = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(lengths, out=new_offsets[1:])

    # Entry k of new deck j comes from old start of that deck + k
    shift = offsets[:-1].astype(np.int64)[order] - new_offsets[:-1]
    entry_order = np.arange(new_offsets[-1], dtype=np.int64)
    entry_order += np.repeat(shift, lengths)
    return new_offsets.astype(np.int32), entry_order


def _sorted_card_uuids(card_lookup: CardLookup, source_game: GameId) -> list[UUID]:
    """source_game's card uuids plus its Unknown sentinel, deduplicated,
    ordered by uuid bytes.

    Private helper - single consumer is build_card_snapshot.
    Inputs: card_lookup, source_game. Output: list[UUID].
    Side effects: none. Exceptions: none.
    """
    card_uuids = set(card_lookup.all_uuids(source_game))
    card_uuids.add(CardBinder.unknown_card_uuid(source_game))
    return sorted(card_uuids, key=lambda card_uuid: card_uuid.bytes)


def _card_table(
    card_uuids: list[UUID], source_game: GameId, stamp: CardSnapshotStamp
) -> pa.Table:
    """The card snapshot table (nocab_uuid, is_unknown), stamp in its
    schema metadata.

    Private helper - single consumer is build_card_snapshot.
    Inputs: card_uuids (ordered), source_game (names the sentinel), stamp.
    Output: pa.Table. Side effects: none. Exceptions: none.
    """
    unknown_uuid = CardBinder.unknown_card_uuid(source_game)
    uuid_column = pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(UUID_WIDTH),
        len(card_uuids),
        [None, pa.py_buffer(b"".join(card_uuid.bytes for card_uuid in card_uuids))],
    )
    is_unknown_column = pa.array(
        [card_uuid == unknown_uuid for card_uuid in card_uuids], type=pa.bool_()
    )
    schema = pa.schema(
        [(NOCAB_UUID_COLUMN, pa.binary(UUID_WIDTH)), (IS_UNKNOWN_COLUMN, pa.bool_())],
        metadata=stamp.to_schema_metadata(),
    )
    return pa.Table.from_arrays([uuid_column, is_unknown_column], schema=schema)


def _require_box_minted_against(
    deck_box: DeckLookup, card_snapshot: CardSnapshot
) -> str:
    """The deck box's version, after checking it was minted against the
    binder card_snapshot was built from.

    Private helper - single consumer is build_deck_snapshot.
    Inputs: deck_box, card_snapshot. Output: str (deck_box.version_for).
    Side effects: reads the box's metadata and deck count.
    Exceptions: StaleDeckBoxError on a missing or different
        card_binder_version.
    """
    source_game = card_snapshot.source_game
    recorded = deck_box.card_binder_version_for(source_game)
    expected = card_snapshot.stamp.card_binder_version
    if recorded != expected:
        raise StaleDeckBoxError(
            f"the {source_game.value} deck box was minted against card binder "
            f"version {recorded!r}, not the current {expected!r}; re-run "
            "scripts/run_deck_box_ingestion.py for the game first"
        )

    # A stamped box always has a version (DeckBox.version_for)
    deck_box_version = deck_box.version_for(source_game)
    if deck_box_version is None:
        raise StaleDeckBoxError(f"the {source_game.value} deck box has no version")
    return deck_box_version
