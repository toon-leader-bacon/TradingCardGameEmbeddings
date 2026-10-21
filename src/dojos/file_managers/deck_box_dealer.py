"""Mechanical train/test/validation wrapper over a DeckBox, backed by
its own small SQLite index file.

The DeckBox counterpart of file_manager_parquet.py: the same "split
management for training consumption" job, on SQLite instead of parquet.
Nothing is held in memory per split; every read is a query.

THE SPLIT ASSIGNMENT comes from the run's HoldoutSpec
(HoldoutSpec.deck_split_of): a pure function of the spec and each deck's
uuid, so every dojo and evaluation built from the same spec agrees on it,
and a deck keeps its split when the box grows. The index file is only a
cache of that function over every deck of the game, so decks_for() and
deck_count() can query a split without holding its uuids in memory.

THE INDEX IS REBUILT at construction when it is empty, or when the spec's
deck_split_key(source_game) or the box's set of source_game deck uuids
differs from what the index records (an index from before the spec split
decks records neither, so it is rebuilt, never reused). The uuid set is
compared by an order-independent digest taken in the same pass as the
count, not by deck content: a deck edited in place keeps its uuid and its
split, so the index stays right.

REPRODUCIBILITY: split MEMBERSHIP is fixed by the spec, and the fixed
order (shuffle=False) is by deck uuid, independent of ingestion order.
Deck uuids are hash-derived (uuid5 of the source id), so effectively
random: the first N decks of a split are a random sample, the same one
every time and whatever the holdout seed. The PER-EPOCH RESHUFFLE
(decks_for(..., shuffle=True), used for TRAIN) is NOT reproducible - it
issues a fresh, unseeded `ORDER BY RANDOM()` per call. Full
reproducibility would mean holding a split's full uuid list in memory,
which this class exists to avoid.
"""

import sqlite3
from pathlib import Path
from typing import Iterator
from uuid import UUID

from src.data_refinement.deck_box.deck_lookup import DeckLookup
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

# One dealer's index file holds exactly one game's assignment (by
# convention - index_path is meant to be per-game, e.g.
# data/splits/contrastive/<dojo>.db), so no source_game column is needed.
# index_info records what the assignment was built from (see
# _INFO_DECK_SPLIT_KEY / _INFO_DECK_SET).
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS deck_splits (
    deck_uuid TEXT PRIMARY KEY,
    split TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deck_splits_split ON deck_splits(split);
CREATE TABLE IF NOT EXISTS index_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_INFO_DECK_SPLIT_KEY = "deck_split_key"
_INFO_DECK_SET = "deck_set"

# The digest of a uuid set: the sum of their 128-bit values, wrapped
_UUID_SET_MODULUS = 2**128

# _assign_splits() batches inserts at this size rather than collecting
# every row before writing - keeping peak memory bounded regardless of
# how many decks a game has.
_ASSIGN_SPLITS_BATCH_SIZE = 10_000


class DeckBoxDealer:
    """Holds one game's train/test/validation deck split (a cache of the
    HoldoutSpec's, in its own SQLite file at index_path) and deals out raw
    deck samples from it. Non-swappable - mechanical plumbing, not a
    research surface.
    """

    def __init__(
        self,
        deck_box: DeckLookup,
        source_game: GameId,
        index_path: Path,
        holdout: HoldoutSpec,
    ) -> None:
        """Connect to (or create) index_path's split index for
        source_game, rebuilding it from holdout if it is missing or stale.

        Inputs:
            deck_box: the deck store to deal from - never queried through
                anything but its public interface.
            source_game: which game's decks to deal. Only this game's
                uuids are considered.
            index_path: this dealer's own SQLite file (separate from
                deck_box's own file). Created if it doesn't exist.
            holdout: the run's spec; holdout.deck_split_of assigns every
                deck's split.
        Output: none (constructor).
        Side effects: creates index_path's parent directory and the file
            if missing; reads every source_game deck uuid in deck_box; rebuilds
            the index (one transaction) when it is empty or stale (see
            this module's docstring).
        Exceptions: UnknownDeckSplitError if holdout has no deck split
            (raised before anything touches disk).

        Example:
            >>> dealer = DeckBoxDealer(
            ...     deck_box, GameId.MTG, Path("data/splits/contrastive/mtg.db"), holdout
            ... )
        """
        # Validate inputs: an unknown deck split raises here, before any I/O
        self._deck_split_key = holdout.deck_split_key(source_game)

        self._deck_box = deck_box
        self._source_game = source_game
        self._holdout = holdout
        self._connection = self._connect_index(index_path)

        # Rebuild the cached assignment unless it matches this spec and box
        deck_set = self._box_deck_set_digest()
        if not self._index_is_current(deck_set):
            self._assign_splits(deck_set)

    @property
    def deck_split_key(self) -> str:
        """The spec's deck_split_key(source_game): everything this
        dealer's split depends on, for caches derived from its decks.

        Inputs: none. Output: str. Side effects: none. Exceptions: none.

        Example:
            >>> cached.deck_split_key == dealer.deck_split_key
            True
        """
        return self._deck_split_key

    @property
    def source_game(self) -> GameId:
        """The game this dealer was constructed for.

        Inputs: none. Output: GameId. Side effects: none.
        Exceptions: none.
        """
        return self._source_game

    @property
    def card_binder_version(self) -> str | None:
        """The CardBinder version self._deck_box's source_game decks
        were minted against, read live - this dealer never caches it,
        since deck_box.card_binder_version_for() is already cheap.

        Inputs: none. Output: str | None (None means untrustworthy -
        see DeckBox.card_binder_version_for()'s own docstring).
        Side effects: none. Exceptions: none.
        """
        return self._deck_box.card_binder_version_for(self._source_game)

    def decks_for(
        self, split: Split, decks_per_sample: int, shuffle: bool = True
    ) -> Iterator[list[GenericDeck]]:
        """Yield any split's decks, grouped into samples.

        Inputs:
            split: which split to deal from.
            decks_per_sample: how many decks per yielded group. Any
                remainder that doesn't fill a full group is dropped.
            shuffle: whether to issue a fresh, freshly-ordered read of
                this split before grouping (e.g. once per epoch for
                TRAIN). NOT reproducible given a seed - see this
                module's docstring's accepted trade-off. True by
                default.
        Output: a generator of lists of GenericDeck, each of length
            decks_per_sample.
        Side effects: none beyond reading from self._deck_box and this
            dealer's own index.
        Exceptions: ValueError if decks_per_sample <= 0.

        Example:
            >>> for deck_sample in dealer.decks_for(Split.TRAIN, 3):
            ...     ...
        """
        if decks_per_sample <= 0:
            raise ValueError("decks_per_sample must be positive")

        # Stream this split's uuids - freshly (re)ordered per call when
        # shuffle=True, a fixed repeatable order otherwise.
        group: list[UUID] = []
        for nocab_uuid in self._stream_split_uuids(split, shuffle):
            group.append(nocab_uuid)
            if len(group) == decks_per_sample:
                yield self._decks_by_uuids(group)
                group = []
        # A trailing partial group is intentionally dropped - every
        # yielded group must have exactly decks_per_sample decks.

    def training_decks(
        self, decks_per_sample: int, shuffle: bool = True
    ) -> Iterator[list[GenericDeck]]:
        """Yield the training split's decks, grouped into samples.

        See decks_for()'s docstring - identical shape, sourced from
        Split.TRAIN.

        Example:
            >>> for deck_sample in dealer.training_decks(decks_per_sample=3):
            ...     ...
        """
        yield from self.decks_for(Split.TRAIN, decks_per_sample, shuffle)

    def test_decks(
        self, decks_per_sample: int, shuffle: bool = True
    ) -> Iterator[list[GenericDeck]]:
        """Yield the test split's decks, grouped into samples.

        See decks_for()'s docstring - identical shape, sourced from
        Split.TEST.
        """
        yield from self.decks_for(Split.TEST, decks_per_sample, shuffle)

    def validation_decks(
        self, decks_per_sample: int, shuffle: bool = True
    ) -> Iterator[list[GenericDeck]]:
        """Yield the validation split's decks, grouped into samples.

        See decks_for()'s docstring - identical shape, sourced from
        Split.VALIDATION.
        """
        yield from self.decks_for(Split.VALIDATION, decks_per_sample, shuffle)

    def deck_count(self, split: Split) -> int:
        """Number of decks assigned to a split.

        Inputs: split. Output: int. Side effects: none. Exceptions: none.

        Example:
            >>> dealer.deck_count(Split.TRAIN)
        """
        cursor = self._connection.execute(
            "SELECT COUNT(*) FROM deck_splits WHERE split = ?", (split.value,)
        )
        return cursor.fetchone()[0]

    # region Private Helpers - construction

    def _connect_index(self, index_path: Path) -> sqlite3.Connection:
        """Connect to (creating, and creating its parent directory and
        schema, if needed) index_path.

        The connection is writable, because __init__ builds the index
        through it, but it is written only during construction. After
        that it is only read, and may be read from any thread, one
        thread at a time (SQLite here is built serialized), so a dealer
        built on one thread can be read by a prefetch worker on another.

        Private helper - single consumer is __init__.
        Inputs: index_path. Output: an open sqlite3.Connection.
        Side effects: creates the parent directory, the file and its tables
        if missing. Exceptions: OSError / sqlite3.Error on failure.
        """
        index_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(index_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.executescript(_SCHEMA_DDL)
        connection.commit()
        return connection

    def _box_deck_set_digest(self) -> str:
        """An order-independent digest of deck_box's source_game deck
        uuids: their count and the sum of their 128-bit values modulo
        _UUID_SET_MODULUS, streamed (never collected).

        Private helper - single consumer is __init__.
        Inputs: none. Output: str ("<count>:<hex sum>").
        Side effects: reads deck_box. Exceptions: none beyond deck_box's
        own.
        """
        count = 0
        total = 0
        for deck_uuid in self._deck_box.all_uuids(self._source_game):
            count += 1
            total = (total + deck_uuid.int) % _UUID_SET_MODULUS
        return f"{count}:{total:032x}"

    def _index_is_current(self, deck_set: str) -> bool:
        """Whether the index holds an assignment built from this spec's
        deck_split_key and the deck uuid set digested as deck_set.

        Private helper - single consumer is __init__. An empty
        deck_splits table is never current, even for an empty box -
        rebuilding nothing is cheap. Safe because _assign_splits() writes
        the rows and index_info in one transaction: a crash leaves the
        previous (stale) state or nothing, never a half-built index
        recorded as current.
        Inputs: deck_set (_box_deck_set_digest's). Output: bool.
        Side effects: reads the index.
        Exceptions: sqlite3.Error on a read failure.
        """
        if (
            self._connection.execute("SELECT 1 FROM deck_splits LIMIT 1").fetchone()
            is None
        ):
            return False
        recorded = {
            row["key"]: row["value"]
            for row in self._connection.execute("SELECT key, value FROM index_info")
        }
        return (
            recorded.get(_INFO_DECK_SPLIT_KEY) == self._deck_split_key
            and recorded.get(_INFO_DECK_SET) == deck_set
        )

    def _assign_splits(self, deck_set: str) -> None:
        """Replace the index's assignment with holdout.deck_split_of over
        every source_game deck, and record what it was built from.

        Private helper - single consumer is __init__. Streams
        deck_box.all_uuids(source_game), batching
        _ASSIGN_SPLITS_BATCH_SIZE rows per INSERT (never the whole game's
        uuids collected first).

        CRASH-SAFETY: the delete, every insert and the index_info write
        run inside one `with self._connection:` transaction, so a raise
        or a killed process rolls back to the previous state - never a
        partial assignment that _index_is_current() could take for a
        complete one.
        Inputs: deck_set (recorded in index_info).
        Output: None.
        Side effects: rewrites deck_splits and index_info.
        Exceptions: sqlite3.Error on a write failure (rolled back).
        """
        batch: list[tuple[str, str]] = []
        with self._connection:
            # Clear the old assignment
            self._connection.execute("DELETE FROM deck_splits")

            # Assign every deck from the spec, in bounded batches
            for deck_uuid in self._deck_box.all_uuids(self._source_game):
                split = self._holdout.deck_split_of(deck_uuid, self._source_game)
                batch.append((str(deck_uuid), split.value))
                if len(batch) >= _ASSIGN_SPLITS_BATCH_SIZE:
                    self._insert_split_batch(batch)
                    batch = []
            if batch:
                self._insert_split_batch(batch)

            # Record what this assignment was built from
            self._write_index_info(deck_set)

    def _write_index_info(self, deck_set: str) -> None:
        """Record deck_split_key and deck_set in index_info. NOT
        committed here - _assign_splits() commits once.

        Private helper - single consumer is _assign_splits().
        Inputs: deck_set. Output: None. Side effects: writes index_info
        (uncommitted). Exceptions: sqlite3.Error on a write failure.
        """
        self._connection.executemany(
            "INSERT OR REPLACE INTO index_info (key, value) VALUES (?, ?)",
            [(_INFO_DECK_SPLIT_KEY, self._deck_split_key), (_INFO_DECK_SET, deck_set)],
        )

    def _insert_split_batch(self, batch: list[tuple[str, str]]) -> None:
        """Insert one batch of (deck_uuid, split) rows into deck_splits.
        NOT committed here - _assign_splits() commits once, after every
        batch has been inserted (see its own docstring's crash-safety
        note).

        Private helper - single consumer is _assign_splits().
        Inputs: batch. Output: None. Side effects: writes deck_splits
        (uncommitted). Exceptions: sqlite3.Error on a write failure.
        """
        self._connection.executemany(
            "INSERT OR REPLACE INTO deck_splits (deck_uuid, split) VALUES (?, ?)",
            batch,
        )

    # endregion Private Helpers - construction

    # region Private Helpers - reading

    def _stream_split_uuids(self, split: Split, shuffle: bool) -> Iterator[UUID]:
        """Every uuid assigned to split, in shuffled or fixed order.

        Private helper - single consumer is decks_for(). shuffle=True
        issues a fresh `ORDER BY RANDOM()` query (see this module's
        docstring's accepted trade-off - NOT reproducible given a
        seed); shuffle=False reads in deck uuid order (fixed and
        independent of ingestion order - see this module's docstring).
        Inputs: split, shuffle. Output: iterator of UUID.
        Side effects: reads the index. Exceptions: sqlite3.Error.
        """
        if shuffle:
            cursor = self._connection.execute(
                "SELECT deck_uuid FROM deck_splits WHERE split = ? ORDER BY RANDOM()",
                (split.value,),
            )
        else:
            cursor = self._connection.execute(
                "SELECT deck_uuid FROM deck_splits WHERE split = ? ORDER BY deck_uuid",
                (split.value,),
            )
        for row in cursor:
            yield UUID(row["deck_uuid"])

    def _decks_by_uuids(self, uuids: list[UUID]) -> list[GenericDeck]:
        """Look up a group of decks by uuid, in order, via one
        self._deck_box.get_by_uuids() call - never a join against
        DeckBox's internal schema.

        Private helper - single consumer is decks_for().
        Inputs: uuids. Output: the decks, in uuids' order.
        Side effects: reads deck_box.
        Exceptions: RuntimeError if a uuid is no longer in deck_box.
        """
        result: list[GenericDeck] = []
        for nocab_uuid, deck in zip(uuids, self._deck_box.get_by_uuids(uuids)):
            if deck is None:
                raise RuntimeError(
                    f"DeckBoxDealer: {nocab_uuid} no longer resolves in "
                    "deck_box - was deck_box mutated after this dealer "
                    "was constructed?"
                )
            result.append(deck)
        return result

    # endregion Private Helpers - reading
