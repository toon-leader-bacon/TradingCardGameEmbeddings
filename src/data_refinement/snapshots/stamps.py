"""The stamps a level-2 snapshot carries in its Arrow schema metadata,
naming the level-1 versions it was compiled from, and the errors raised
when a snapshot or its deck box no longer matches.

A stamp round-trips through schema metadata (to_schema_metadata /
from_schema_metadata) so every reader compares a typed value, never raw
metadata bytes. A deck snapshot's metadata holds its card snapshot's
keys too, so one file names both versions it depends on.
"""

from dataclasses import dataclass
from pathlib import Path

# Every level-2 file lives under here (ignored by git, rebuilt, never edited)
SNAPSHOT_ROOT = Path("data/derived")

# Bumped when a snapshot's columns or their meaning change; a file with
# another format_version is stale
CARD_SNAPSHOT_FORMAT_VERSION = 1
DECK_SNAPSHOT_FORMAT_VERSION = 1

# The command every stale-snapshot message names
BUILD_COMMAND = "PYTHONPATH=. python3 scripts/build_training_snapshots.py --game <game>"

# Schema metadata keys. Card and deck keys differ, so a deck snapshot's
# metadata holds both stamps flat
_CARD_BINDER_VERSION_KEY = b"card_binder_version"
_CARD_FORMAT_VERSION_KEY = b"card_snapshot_format_version"
_DECK_BOX_VERSION_KEY = b"deck_box_version"
_DECK_FORMAT_VERSION_KEY = b"deck_snapshot_format_version"


class StaleSnapshotError(Exception):
    """A level-2 file is missing or its stamp no longer matches level 1.
    The message names the build command."""


class StaleDeckBoxError(Exception):
    """The deck box was minted against another binder (or never stamped),
    so its cards cannot be indexed into the card snapshot."""


@dataclass(frozen=True)
class CardSnapshotStamp:
    """What a card snapshot was compiled from.

    card_binder_version: card_lookup.version_for(game) at build.
    format_version: CARD_SNAPSHOT_FORMAT_VERSION at build.
    """

    card_binder_version: str
    format_version: int

    def to_schema_metadata(self) -> dict[bytes, bytes]:
        """This stamp as Arrow schema metadata.

        Inputs: none. Output: dict[bytes, bytes].
        Side effects: none. Exceptions: none.

        Example:
            >>> schema.with_metadata(stamp.to_schema_metadata())
        """
        return {
            _CARD_BINDER_VERSION_KEY: self.card_binder_version.encode(),
            _CARD_FORMAT_VERSION_KEY: str(self.format_version).encode(),
        }

    @classmethod
    def from_schema_metadata(
        cls, metadata: dict[bytes, bytes] | None
    ) -> "CardSnapshotStamp":
        """Read a stamp back from Arrow schema metadata.

        Inputs: metadata (a file's schema.metadata; None when it has none).
        Output: CardSnapshotStamp.
        Side effects: none.
        Exceptions: StaleSnapshotError if a key is missing or unreadable.

        Example:
            >>> CardSnapshotStamp.from_schema_metadata(table.schema.metadata)
        """
        return cls(
            card_binder_version=_text_of(metadata, _CARD_BINDER_VERSION_KEY),
            format_version=_integer_of(metadata, _CARD_FORMAT_VERSION_KEY),
        )


@dataclass(frozen=True)
class DeckSnapshotStamp:
    """What a deck snapshot was compiled from.

    card_snapshot: the stamp of the card snapshot its indices point into.
    deck_box_version: deck_box.version_for(game) at build.
    format_version: DECK_SNAPSHOT_FORMAT_VERSION at build.
    """

    card_snapshot: CardSnapshotStamp
    deck_box_version: str
    format_version: int

    def to_schema_metadata(self) -> dict[bytes, bytes]:
        """This stamp (with its card snapshot stamp's keys) as Arrow schema
        metadata.

        Inputs: none. Output: dict[bytes, bytes].
        Side effects: none. Exceptions: none.

        Example:
            >>> schema.with_metadata(stamp.to_schema_metadata())
        """
        return {
            **self.card_snapshot.to_schema_metadata(),
            _DECK_BOX_VERSION_KEY: self.deck_box_version.encode(),
            _DECK_FORMAT_VERSION_KEY: str(self.format_version).encode(),
        }

    @classmethod
    def from_schema_metadata(
        cls, metadata: dict[bytes, bytes] | None
    ) -> "DeckSnapshotStamp":
        """Read a stamp back from Arrow schema metadata.

        Inputs: metadata (a file's schema.metadata; None when it has none).
        Output: DeckSnapshotStamp.
        Side effects: none.
        Exceptions: StaleSnapshotError if a key is missing or unreadable.

        Example:
            >>> DeckSnapshotStamp.from_schema_metadata(table.schema.metadata)
        """
        return cls(
            card_snapshot=CardSnapshotStamp.from_schema_metadata(metadata),
            deck_box_version=_text_of(metadata, _DECK_BOX_VERSION_KEY),
            format_version=_integer_of(metadata, _DECK_FORMAT_VERSION_KEY),
        )


def _text_of(metadata: dict[bytes, bytes] | None, key: bytes) -> str:
    """metadata[key] decoded as UTF-8.

    Private helper - shared by both stamps' from_schema_metadata.
    Inputs: metadata, key. Output: str. Side effects: none.
    Exceptions: StaleSnapshotError if the key is missing or not UTF-8.
    """
    if metadata is None or key not in metadata:
        raise StaleSnapshotError(
            f"snapshot metadata has no {key.decode()} stamp; rebuild with "
            f"{BUILD_COMMAND}"
        )
    try:
        return metadata[key].decode("utf-8")
    except UnicodeDecodeError as error:
        raise StaleSnapshotError(
            f"snapshot metadata {key.decode()} is unreadable; rebuild with "
            f"{BUILD_COMMAND}"
        ) from error


def _integer_of(metadata: dict[bytes, bytes] | None, key: bytes) -> int:
    """metadata[key] as an int.

    Private helper - shared by both stamps' from_schema_metadata.
    Inputs: metadata, key. Output: int. Side effects: none.
    Exceptions: StaleSnapshotError if the key is missing or not an int.
    """
    text = _text_of(metadata, key)
    try:
        return int(text)
    except ValueError as error:
        raise StaleSnapshotError(
            f"snapshot metadata {key.decode()} is not an integer ({text!r}); "
            f"rebuild with {BUILD_COMMAND}"
        ) from error
