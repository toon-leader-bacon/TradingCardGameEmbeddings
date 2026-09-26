"""Row-level SQL access to one DeckBox connection's tables.

Private to deck_box.py: DeckTables is a Table Data Gateway (one method
per SQL statement DeckBox needs) so DeckBox itself only decides WHAT
to write and WHEN to commit, never HOW a row is spelled. Nothing here
commits - every write joins whatever `with connection:` block its
caller has open (see DeckBox's crash-safety docstring).
"""

import hashlib
import secrets
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterator
from uuid import UUID

from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

# decks/deck_cards/metadata. IF NOT EXISTS throughout: create_schema()
# runs on every connection DeckBox opens, including one already holding
# this schema from a prior run.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS decks (
    nocab_uuid TEXT PRIMARY KEY,
    source_game TEXT NOT NULL,
    name TEXT NOT NULL,
    provenance_data_source TEXT,
    provenance_source_id TEXT,
    provenance_fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS deck_cards (
    deck_uuid TEXT NOT NULL REFERENCES decks(nocab_uuid),
    card_uuid TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (deck_uuid, card_uuid)
);
CREATE INDEX IF NOT EXISTS idx_deck_cards_deck_uuid ON deck_cards(deck_uuid);

CREATE TABLE IF NOT EXISTS metadata (
    source_game TEXT PRIMARY KEY,
    card_binder_version TEXT NOT NULL
);
"""

# Name register_seeded_order() registers on a connection and
# ranked_deck_rows() sorts by - see DeckBox.uuids_ranked_randomly()'s
# docstring for why this exists instead of SQLite's own RANDOM().
_SEEDED_ORDER_FUNCTION_NAME = "deckbox_seeded_order"

_UPSERT_METADATA_SQL = (
    "INSERT OR REPLACE INTO metadata (source_game, card_binder_version) VALUES (?, ?)"
)


def stamp_metadata_at(
    path: Path, source_game: GameId, card_binder_version: str
) -> None:
    """Insert or overwrite source_game's metadata row directly in path's
    own file, via a short-lived separate connection.

    Inputs:
        path: an existing DeckBox SQLite file.
        source_game: which game's row to write.
        card_binder_version: the version to stamp.
    Output: none.
    Side effects: writes and commits one row in path.
    Exceptions: sqlite3.Error if path isn't a valid DeckBox file.
    """
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            _UPSERT_METADATA_SQL, (source_game.value, card_binder_version)
        )
        connection.commit()
    finally:
        connection.close()


def deck_from_row(row: sqlite3.Row, card_nocab_uuids: list[UUID]) -> GenericDeck:
    """Reconstruct a GenericDeck from a decks row plus its already-
    fetched card_nocab_uuids.

    Inputs:
        row: one decks row (sqlite3.Row).
        card_nocab_uuids: that deck's expanded card multiset.
    Output: GenericDeck.
    Side effects: none.
    Exceptions: ValueError if a stored enum value is unknown.
    """
    provenance = None
    if row["provenance_data_source"] is not None:
        provenance = Provenance(
            data_source=DataSource(row["provenance_data_source"]),
            source_id=row["provenance_source_id"],
            fetched_at=datetime.fromisoformat(row["provenance_fetched_at"]),
        )
    return GenericDeck(
        nocab_uuid=UUID(row["nocab_uuid"]),
        source_game=GameId(row["source_game"]),
        name=row["name"],
        card_nocab_uuids=card_nocab_uuids,
        provenance=provenance,
    )


def _deck_row_values(deck: GenericDeck) -> tuple:
    """deck's own-row column values, in decks' column order - shared by
    insert_deck_row()/update_deck_row() so the two never drift apart on
    which columns exist."""
    provenance = deck.provenance
    return (
        str(deck.nocab_uuid),
        deck.source_game.value,
        deck.name,
        provenance.data_source.value if provenance is not None else None,
        provenance.source_id if provenance is not None else None,
        provenance.fetched_at.isoformat() if provenance is not None else None,
    )


class DeckTables:
    """Every SQL statement DeckBox runs against its decks/deck_cards/
    metadata tables, on one connection. Holds no state of its own
    beyond that connection; never commits.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        """
        Inputs: connection, the sqlite3 connection to read and write.
        Output: none (constructor).
        Side effects: none (call create_schema() to set the connection up).
        Exceptions: none.
        """
        self._connection = connection

    def create_schema(self) -> None:
        """Set row_factory and create decks/deck_cards/metadata if absent.

        Inputs: none. Output: none.
        Side effects: executes the schema DDL and commits.
        Exceptions: sqlite3.Error on a corrupt database.
        """
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA_DDL)
        self._connection.commit()

    # region metadata

    def clear_metadata(self, source_game: GameId) -> None:
        """Delete source_game's metadata row (a no-op if absent).

        Inputs: source_game. Output: none.
        Side effects: one DELETE, uncommitted. Exceptions: none.
        """
        self._connection.execute(
            "DELETE FROM metadata WHERE source_game = ?", (source_game.value,)
        )

    def metadata_row(self, source_game: GameId) -> sqlite3.Row | None:
        """The metadata row for source_game, or None if absent.

        Inputs: source_game. Output: sqlite3.Row | None.
        Side effects: none. Exceptions: none.
        """
        cursor = self._connection.execute(
            "SELECT * FROM metadata WHERE source_game = ?", (source_game.value,)
        )
        return cursor.fetchone()

    def upsert_metadata(self, source_game: GameId, card_binder_version: str) -> None:
        """Insert or overwrite source_game's metadata row.

        Inputs: source_game, card_binder_version. Output: none.
        Side effects: one write, uncommitted. Exceptions: none.
        """
        self._connection.execute(
            _UPSERT_METADATA_SQL, (source_game.value, card_binder_version)
        )

    # endregion metadata

    # region reads

    def deck_exists(self, nocab_uuid: UUID) -> bool:
        """Whether nocab_uuid has a row in decks.

        Inputs: nocab_uuid. Output: bool.
        Side effects: none. Exceptions: none.
        """
        cursor = self._connection.execute(
            "SELECT 1 FROM decks WHERE nocab_uuid = ?", (str(nocab_uuid),)
        )
        return cursor.fetchone() is not None

    def deck_row(self, nocab_uuid: UUID) -> sqlite3.Row | None:
        """The decks row for nocab_uuid, or None if absent.

        Inputs: nocab_uuid. Output: sqlite3.Row | None.
        Side effects: none. Exceptions: none.
        """
        cursor = self._connection.execute(
            "SELECT * FROM decks WHERE nocab_uuid = ?", (str(nocab_uuid),)
        )
        return cursor.fetchone()

    def deck_cards(self, nocab_uuid: UUID) -> list[UUID]:
        """nocab_uuid's card multiset, expanded from deck_cards'
        (card_uuid, count) rows into a flat, duplicates-meaningful list.

        Inputs: nocab_uuid. Output: list[UUID].
        Side effects: none. Exceptions: none.
        """
        cursor = self._connection.execute(
            "SELECT card_uuid, count FROM deck_cards WHERE deck_uuid = ?",
            (str(nocab_uuid),),
        )
        card_nocab_uuids: list[UUID] = []
        for row in cursor.fetchall():
            card_nocab_uuids.extend([UUID(row["card_uuid"])] * row["count"])
        return card_nocab_uuids

    def deck_rows(
        self, source_game: GameId | None, *, ordered_by_uuid: bool = False
    ) -> Iterator[sqlite3.Row]:
        """Every decks row, filtered by source_game if given, lazily.

        Inputs:
            source_game: filter, or None for every game.
            ordered_by_uuid: sort by nocab_uuid (for content hashing);
                otherwise no guaranteed order.
        Output: Iterator[sqlite3.Row].
        Side effects: none. Exceptions: none.
        """
        sql = "SELECT * FROM decks"
        params: tuple = ()
        if source_game is not None:
            sql += " WHERE source_game = ?"
            params = (source_game.value,)
        if ordered_by_uuid:
            sql += " ORDER BY nocab_uuid"
        yield from self._connection.execute(sql, params)

    def ranked_deck_rows(
        self, source_game: GameId, seed: int | None
    ) -> Iterator[sqlite3.Row]:
        """Every source_game deck's (nocab_uuid, rank, total), ranked
        1..total by a seeded deterministic order computed inside SQLite.

        Inputs:
            source_game: which game's decks to rank.
            seed: ordering seed; None draws one fresh random seed for
                this call (one coherent permutation, not per-row noise).
        Output: Iterator[sqlite3.Row] with nocab_uuid/rank/total columns.
        Side effects: registers a SQL function on the connection.
        Exceptions: none.
        """
        self._register_seeded_order(seed)
        cursor = self._connection.execute(
            f"""
            SELECT nocab_uuid,
                   ROW_NUMBER() OVER (ORDER BY {_SEEDED_ORDER_FUNCTION_NAME}(nocab_uuid)) AS rank,
                   COUNT(*) OVER () AS total
            FROM decks
            WHERE source_game = ?
            """,
            (source_game.value,),
        )
        yield from cursor

    def _register_seeded_order(self, seed: int | None) -> None:
        """Register sha256(seed:nocab_uuid) as a SQL ordering function -
        NOT SQLite's RANDOM(), which cannot be seeded."""
        effective_seed = seed if seed is not None else secrets.token_hex(16)

        def seeded_order(nocab_uuid: str) -> str:
            return hashlib.sha256(f"{effective_seed}:{nocab_uuid}".encode()).hexdigest()

        self._connection.create_function(_SEEDED_ORDER_FUNCTION_NAME, 1, seeded_order)

    # endregion reads

    # region writes (never committed here)

    def insert_deck_row(self, deck: GenericDeck) -> None:
        """Insert deck's own decks row (not its deck_cards rows).

        Inputs: deck. Output: none.
        Side effects: one INSERT, uncommitted.
        Exceptions: sqlite3.IntegrityError on a duplicate nocab_uuid.
        """
        self._connection.execute(
            """
            INSERT INTO decks (
                nocab_uuid, source_game, name,
                provenance_data_source, provenance_source_id, provenance_fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            _deck_row_values(deck),
        )

    def update_deck_row(self, deck: GenericDeck) -> None:
        """Overwrite an already-stored deck's own decks row.

        Inputs: deck. Output: none.
        Side effects: one UPDATE, uncommitted. Exceptions: none.
        """
        nocab_uuid, source_game, name, data_source, source_id, fetched_at = (
            _deck_row_values(deck)
        )
        self._connection.execute(
            """
            UPDATE decks
            SET source_game = ?, name = ?,
                provenance_data_source = ?, provenance_source_id = ?, provenance_fetched_at = ?
            WHERE nocab_uuid = ?
            """,
            (source_game, name, data_source, source_id, fetched_at, nocab_uuid),
        )

    def delete_deck_row(self, nocab_uuid: UUID) -> None:
        """Delete nocab_uuid's decks row.

        Inputs: nocab_uuid. Output: none.
        Side effects: one DELETE, uncommitted. Exceptions: none.
        """
        self._connection.execute(
            "DELETE FROM decks WHERE nocab_uuid = ?", (str(nocab_uuid),)
        )

    def insert_deck_cards(self, deck_uuid: UUID, card_nocab_uuids: list[UUID]) -> None:
        """Insert deck_uuid's card multiset into deck_cards, one
        (deck_uuid, card_uuid, count) row per distinct card.

        Inputs: deck_uuid, card_nocab_uuids. Output: none.
        Side effects: INSERTs, uncommitted. Exceptions: none.
        """
        counts = Counter(card_nocab_uuids)
        self._connection.executemany(
            "INSERT INTO deck_cards (deck_uuid, card_uuid, count) VALUES (?, ?, ?)",
            [
                (str(deck_uuid), str(card_uuid), count)
                for card_uuid, count in counts.items()
            ],
        )

    def replace_deck_cards(self, deck_uuid: UUID, card_nocab_uuids: list[UUID]) -> None:
        """Replace deck_uuid's deck_cards rows wholesale.

        Inputs: deck_uuid, card_nocab_uuids. Output: none.
        Side effects: DELETE + INSERTs, uncommitted. Exceptions: none.
        """
        self.delete_deck_cards(deck_uuid)
        self.insert_deck_cards(deck_uuid, card_nocab_uuids)

    def delete_deck_cards(self, deck_uuid: UUID) -> None:
        """Delete every deck_cards row for deck_uuid.

        Inputs: deck_uuid. Output: none.
        Side effects: one DELETE, uncommitted. Exceptions: none.
        """
        self._connection.execute(
            "DELETE FROM deck_cards WHERE deck_uuid = ?", (str(deck_uuid),)
        )

    def merge_from(self, path: Path) -> None:
        """Read-only, last-path-wins merge of path's decks/deck_cards/
        metadata into this connection. A deck path also provides has
        its existing deck_cards rows cleared first, since INSERT OR
        REPLACE into decks alone wouldn't remove card rows path's
        version doesn't have.

        Inputs: path, a DeckBox SQLite file. Output: none.
        Side effects: ATTACHes path read-only, writes and commits into
            this connection, DETACHes. Never writes to path.
        Exceptions: sqlite3.Error if path isn't a valid DeckBox file.
        """
        self._connection.execute("ATTACH DATABASE ? AS merge_source", (str(path),))
        try:
            self._connection.execute(
                "DELETE FROM deck_cards WHERE deck_uuid IN "
                "(SELECT nocab_uuid FROM merge_source.decks)"
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO decks SELECT * FROM merge_source.decks"
            )
            self._connection.execute(
                "INSERT INTO deck_cards SELECT * FROM merge_source.deck_cards"
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata SELECT * FROM merge_source.metadata"
            )
            self._connection.commit()
        finally:
            self._connection.execute("DETACH DATABASE merge_source")

    # endregion writes
