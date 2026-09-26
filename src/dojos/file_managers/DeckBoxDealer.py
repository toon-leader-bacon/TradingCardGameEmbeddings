"""Mechanical train/test/validation wrapper over a DeckBox, backed by
its own small SQLite index file.

See plans/deckbox_sqlite.md's "Rewriting DeckBoxDealer" section for the
full design this implements. RELOCATED here from
src/dojos/contrastive/deck_box_dealer.py, where it originally lived -
it was never conceptually contrastive-specific, and this repo already
reserves src/dojos/file_managers/ for exactly this shape of component
(FileManagerCSV, FileManagerParquet). Parallel in role to those two:
same "split management for training consumption" job, on a different
storage technology.

REWRITTEN from an eager, fully in-memory design (load every uuid for a
game into one Python list, shuffle/partition it, hold all three lists
for the instance's lifetime) onto SQLite - the old design's only real
limitation was that eager materialization, not a permanent difference
in need from any other DeckBox consumer; a single, uniform
implementation now serves every scale.

THE SPLIT ASSIGNMENT is exact-ratio and computed ONCE, at construction
(unless index_path already holds one for source_game and
force_resplit isn't set): this class streams DeckBox.uuids_ranked_randomly()'s
already-ranked (uuid, rank, total) tuples and batches them into this
class's own deck_splits table - a Python loop consuming an SQL-native
ranking, NOT a single cross-database SQL statement (that would need
ATTACH DATABASE, meaning this class would have to know DeckBox's file
path directly - the exact coupling uuids_ranked_randomly() exists to
avoid).

ASSUMPTION THIS RELIES ON: a game's DeckBox is not expected to grow
AFTER this assignment has run for it - training happens once ingestion
is complete, not concurrently with it. A growing box after assignment
would make the fixed "80% by rank" boundary drift; nothing in this
codebase has that shape today.

A REAL, ACCEPTED TRADE-OFF: the one-time split ASSIGNMENT is seeded and
reproducible (via uuids_ranked_randomly()'s own deterministic
ordering), but the PER-EPOCH RESHUFFLE (decks_for(..., shuffle=True),
used for TRAIN) is NOT - it issues a fresh, unseeded `ORDER BY RANDOM()`
per call, so two runs with the same seed get identical split
MEMBERSHIP but a DIFFERENT per-epoch read ORDER. The old in-memory
implementation was fully reproducible end-to-end (one seeded
random.Random instance advancing deterministically across every
reshuffle call). This was discussed and accepted: chasing full
reproducibility here would mean holding a split's full uuid list in
memory again, which is exactly what this rewrite exists to avoid.
"""

import sqlite3
from pathlib import Path
from typing import Iterator
from uuid import UUID

from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.splits import Split

# One dealer's index file holds exactly one game's assignment (by
# convention - index_path is meant to be per-game, e.g.
# data/splits/mtg_dealer.db), so no source_game column is needed here.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS deck_splits (
    deck_uuid TEXT PRIMARY KEY,
    split TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deck_splits_split ON deck_splits(split);
"""

# _assign_splits() batches inserts at this size rather than collecting
# every row before writing - keeping peak memory bounded regardless of
# how many decks a game has, the whole point of this rewrite.
_ASSIGN_SPLITS_BATCH_SIZE = 10_000


class DeckBoxDealer:
    """Owns one game's train/test/validation split assignment (in its
    own SQLite file at index_path) and deals out raw deck samples from
    it. Non-swappable - mechanical plumbing, not a research surface.
    """

    def __init__(
        self,
        deck_box: DeckBox,
        source_game: GameId,
        index_path: Path,
        split_ratios: list[float] = [8, 1, 1],
        seed: int | None = None,
        force_resplit: bool = False,
    ) -> None:
        """Connect to (or create) index_path's split assignment for
        source_game, computing it if it doesn't already exist.

        Inputs:
            deck_box: the deck store to deal from and to rank via
                uuids_ranked_randomly() - never queried through
                anything but its public interface.
            source_game: which game's decks to partition. Only this
                game's uuids are considered.
            index_path: this dealer's own SQLite file (separate from
                deck_box's own file), holding the persisted
                deck_splits assignment. Created if it doesn't exist.
            split_ratios: relative train/test/validation sizes, in
                that order (e.g. [8, 1, 1] is an 80/10/10 split,
                exact - see this module's docstring). Need not sum to
                any particular total.
            seed: seed for the one-time split assignment (see
                uuids_ranked_randomly()). Reproducible given the same
                seed and box content; does NOT make the per-epoch
                reshuffle in decks_for() reproducible - see this
                module's docstring's accepted trade-off.
            force_resplit: recompute the split assignment even if one
                already exists at index_path for source_game. False
                (default) skips recomputation when one is already
                there - mirrors DojoConfig's own force_resplit escape
                hatch.
        Output: none (constructor).
        Side effects: creates index_path's parent directory and the
            file itself if missing; if no assignment already exists
            for source_game (or force_resplit is True), streams every
            source_game uuid from deck_box and writes this game's full
            split assignment to index_path.
        Exceptions: ValueError if split_ratios doesn't have exactly 3
            entries, or any entry is negative, or they sum to 0.

        Example:
            >>> dealer = DeckBoxDealer(
            ...     deck_box, GameId.MTG, Path("data/splits/mtg_dealer.db"), seed=42
            ... )
        """
        self._validate_split_ratios(split_ratios)

        self._deck_box = deck_box
        self._source_game = source_game
        self._split_ratios = split_ratios
        self._seed = seed
        self._connection = self._connect_index(index_path)

        if force_resplit or not self._splits_exist():
            self._assign_splits()

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

    def _validate_split_ratios(self, split_ratios: list[float]) -> None:
        """Raise ValueError if split_ratios isn't exactly 3
        non-negative entries summing to a positive value.

        Private helper - single consumer is __init__.
        """
        if len(split_ratios) != 3:
            raise ValueError(
                f"split_ratios must have exactly 3 entries, got {len(split_ratios)}"
            )
        if any(ratio < 0 for ratio in split_ratios):
            raise ValueError("split_ratios entries must be non-negative")
        if sum(split_ratios) <= 0:
            raise ValueError("split_ratios must sum to a positive value")

    def _connect_index(self, index_path: Path) -> sqlite3.Connection:
        """Connect to (creating, and creating its parent directory and
        deck_splits schema, if needed) index_path.

        Private helper - single consumer is __init__.
        """
        index_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(index_path))
        connection.row_factory = sqlite3.Row
        connection.executescript(_SCHEMA_DDL)
        connection.commit()
        return connection

    def _splits_exist(self) -> bool:
        """Whether this dealer's index already holds a split
        assignment for self._source_game.

        Private helper - single consumer is __init__. index_path is a
        per-game file by convention (see this module's schema
        comment), so "any row at all" is equivalent to "already
        assigned for this game." This simple check is only safe
        because _assign_splits() commits the WHOLE assignment in one
        transaction (see its own docstring) - a crash partway through
        assigning leaves deck_splits with zero rows, not a partial
        assignment that this check would otherwise have to distinguish
        from a complete one.
        """
        cursor = self._connection.execute("SELECT 1 FROM deck_splits LIMIT 1")
        return cursor.fetchone() is not None

    def _assign_splits(self) -> None:
        """Compute and persist an exact-ratio split assignment for
        self._source_game, from scratch, into this dealer's own
        deck_splits table.

        Private helper - single consumer is __init__. Streams
        self._deck_box.uuids_ranked_randomly(self._source_game,
        self._seed)'s (uuid, rank, total) tuples; for each, compares
        rank/total against self._split_ratios' cumulative boundaries to
        decide train/test/validation, and batches the resulting rows
        into an INSERT here (_ASSIGN_SPLITS_BATCH_SIZE at a time, never
        the whole game's uuids collected first - see this module's
        docstring for why this is a streamed Python loop over an
        already-SQL-ranked source, not a single cross-database
        statement).

        CRASH-SAFETY: the whole loop runs inside `with self._connection:`,
        which commits ONCE on a clean return and rolls back everything
        inserted so far if this raises OR if the process is killed
        before it finishes - either way, deck_splits ends up with zero
        rows for this game, never a partial assignment that
        _splits_exist() could mistake for a complete one. (An explicit
        rollback matters here, not just "a killed process discards an
        uncommitted transaction": without it, a caller that catches an
        exception from this and retries against the same index_path
        would otherwise be racing this connection's own eventual
        garbage-collection-triggered close() to release the write
        lock - `with self._connection:` makes the rollback immediate
        and synchronous instead.) This also mirrors why bulk-inserting
        in one transaction is the standard technique for making it
        fast, not only for making it safe.
        """
        boundaries = self._cumulative_ratio_boundaries()
        batch: list[tuple[str, str]] = []
        with self._connection:
            for nocab_uuid, rank, total in self._deck_box.uuids_ranked_randomly(
                self._source_game, self._seed
            ):
                split = self._split_for_rank(rank, total, boundaries)
                batch.append((str(nocab_uuid), split.value))
                if len(batch) >= _ASSIGN_SPLITS_BATCH_SIZE:
                    self._insert_split_batch(batch)
                    batch = []
            if batch:
                self._insert_split_batch(batch)

    def _cumulative_ratio_boundaries(self) -> tuple[float, float]:
        """self._split_ratios as cumulative (train_end, test_end)
        fractions of the whole - e.g. [8, 1, 1] -> (0.8, 0.9).

        Private helper - single consumer is _assign_splits().
        """
        total_ratio = sum(self._split_ratios)
        train_end = self._split_ratios[0] / total_ratio
        test_end = train_end + self._split_ratios[1] / total_ratio
        return train_end, test_end

    def _split_for_rank(
        self, rank: int, total: int, boundaries: tuple[float, float]
    ) -> Split:
        """Which split rank/total (1-indexed rank out of total) falls
        into, given cumulative (train_end, test_end) boundaries.

        Private helper - single consumer is _assign_splits().
        """
        train_end, test_end = boundaries
        fraction = rank / total
        if fraction <= train_end:
            return Split.TRAIN
        if fraction <= test_end:
            return Split.TEST
        return Split.VALIDATION

    def _insert_split_batch(self, batch: list[tuple[str, str]]) -> None:
        """Insert one batch of (deck_uuid, split) rows into deck_splits.
        NOT committed here - _assign_splits() commits once, after every
        batch has been inserted (see its own docstring's crash-safety
        note).

        Private helper - single consumer is _assign_splits().
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
        seed); shuffle=False reads in a fixed, repeatable order.
        """
        if shuffle:
            cursor = self._connection.execute(
                "SELECT deck_uuid FROM deck_splits WHERE split = ? ORDER BY RANDOM()",
                (split.value,),
            )
        else:
            cursor = self._connection.execute(
                "SELECT deck_uuid FROM deck_splits WHERE split = ?", (split.value,)
            )
        for row in cursor:
            yield UUID(row["deck_uuid"])

    def _decks_by_uuids(self, uuids: list[UUID]) -> list[GenericDeck]:
        """Look up a group of decks by uuid, in order, via
        self._deck_box.get_by_uuid() - never a join against DeckBox's
        internal schema.

        Private helper - single consumer is decks_for().
        """
        result: list[GenericDeck] = []
        for nocab_uuid in uuids:
            deck = self._deck_box.get_by_uuid(nocab_uuid)
            if deck is None:
                raise RuntimeError(
                    f"DeckBoxDealer: {nocab_uuid} no longer resolves in "
                    "deck_box - was deck_box mutated after this dealer "
                    "was constructed?"
                )
            result.append(deck)
        return result

    # endregion Private Helpers - reading
