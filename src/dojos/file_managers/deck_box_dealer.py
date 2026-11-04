"""Mechanical train/test/validation dealing over one game's deck snapshot.

The deck-snapshot counterpart of file_manager_parquet.py: the same "split
management for training consumption" job, over level-2 Arrow arrays
(src/data_refinement/snapshots/) instead of SQLite queries.

THE SPLIT ASSIGNMENT is the run's DeckSplitIndex: HoldoutSpec.deck_split_of
over every deck of the snapshot, a pure function of the spec and each
deck's uuid, so every dojo and evaluation built from the same spec agrees
on it, and a deck keeps its split when the box grows.

REPRODUCIBILITY: split membership is fixed by the spec. The fixed order
(shuffle=False, used for TEST and VALIDATION) is deck uuid order. Deck
uuids are hash-derived, so effectively random: the first N decks of a
split are a random sample, the same one every time. The per-epoch
reshuffle (shuffle=True, used for TRAIN) is a NumPy permutation of the
split's rows drawn from a generator seeded at construction, so a run's
sequence of epochs is reproducible. The split's rows are held in memory
to do that: an int64 row array is about 44 MB for MTG's TRAIN split.

A dealt deck is a GenericDeck rebuilt from its snapshot row: its cards
expanded by count, in card-index order (a multiset, so order carries no
meaning), name "" and provenance None (the snapshot keeps neither; no
contrastive consumer reads them).
"""

from typing import Iterator
from uuid import UUID

import numpy as np

from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.dojos.file_managers.deck_split_index import DeckSplitIndex
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.splits import Split


class DeckBoxDealer:
    """Deals one game's decks, split by a DeckSplitIndex, as raw deck
    samples. Non-swappable - mechanical plumbing, not a research surface.
    Safe to deal from one thread at a time (the prefetch worker), whichever
    thread built it.
    """

    def __init__(
        self, snapshot: DeckSnapshot, split_index: DeckSplitIndex, shuffle_seed: int
    ) -> None:
        """Deal from snapshot's decks, split by split_index.

        Inputs:
            snapshot: the game's open deck snapshot.
            split_index: the run's split over snapshot's rows.
            shuffle_seed: seeds the per-epoch TRAIN reshuffle.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError if split_index numbers another snapshot's
            rows (its snapshot_stamp differs from snapshot.stamp).

        Example:
            >>> dealer = DeckBoxDealer(decks, split_index, shuffle_seed=0)
        """
        # Validate inputs: the split's row numbers must be this snapshot's
        if split_index.snapshot_stamp != snapshot.stamp:
            raise ValueError(
                "DeckBoxDealer: the split index numbers the rows of another "
                f"{snapshot.source_game.value} deck snapshot; rebuild it with "
                "DeckSplitIndex.load_or_build"
            )

        self._snapshot = snapshot
        self._split_index = split_index
        self._shuffle_rng = np.random.default_rng(shuffle_seed)

    @property
    def deck_split_key(self) -> str:
        """The spec's deck_split_key(source_game): everything this
        dealer's split depends on, for caches derived from its decks.

        Inputs: none. Output: str. Side effects: none. Exceptions: none.

        Example:
            >>> cached.deck_split_key == dealer.deck_split_key
            True
        """
        return self._split_index.deck_split_key

    @property
    def source_game(self) -> GameId:
        """The snapshot's game.

        Inputs: none. Output: GameId. Side effects: none.
        Exceptions: none.
        """
        return self._snapshot.source_game

    @property
    def card_binder_version(self) -> str | None:
        """The CardBinder version the dealt decks were minted against: the
        card snapshot's card_binder_version, which equals the deck box's
        (build_deck_snapshot refuses a box minted against another binder).

        Inputs: none. Output: str | None (always a str here; the type
            keeps ContrastiveDojo's check unchanged).
        Side effects: none. Exceptions: none.
        """
        return self._snapshot.stamp.card_snapshot.card_binder_version

    def decks_for(
        self,
        split: Split,
        decks_per_sample: int,
        shuffle: bool = True,
        keep_remainder: bool = False,
    ) -> Iterator[list[GenericDeck]]:
        """Yield any split's decks, grouped into samples.

        Inputs:
            split: which split to deal from.
            decks_per_sample: how many decks per yielded group.
            shuffle: whether to deal this pass in a fresh permutation
                drawn from the dealer's seeded generator (e.g. once per
                epoch for TRAIN), or in deck uuid order. True by default.
            keep_remainder: whether the split's last decks, too few to
                fill a group, are yielded as one shorter final group
                (True) or dropped (False, the default).
        Output: a generator of lists of GenericDeck, each of length
            decks_per_sample, except a shorter last one when
            keep_remainder is True.
        Side effects: a shuffled pass advances the shuffle generator
            (when the generator is first advanced).
        Exceptions: ValueError if decks_per_sample <= 0.

        Example:
            >>> for deck_sample in dealer.decks_for(Split.TRAIN, 3):
            ...     ...
        """
        if decks_per_sample <= 0:
            raise ValueError("decks_per_sample must be positive")

        # This pass's row order
        rows = self._split_index.rows_of(split)
        if shuffle:
            # Permute positions: rows itself is the index's read-only array
            rows = rows[self._shuffle_rng.permutation(len(rows))]

        # Deal full groups, then the remainder if asked
        for start in range(0, len(rows), decks_per_sample):
            group = rows[start : start + decks_per_sample]
            if len(group) < decks_per_sample and not keep_remainder:
                return
            yield [self._deck_at(row) for row in group.tolist()]

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
        return len(self._split_index.rows_of(split))

    def _deck_at(self, row: int) -> GenericDeck:
        """The snapshot's deck at row as a GenericDeck: its card indices
        repeated by count and mapped through the card snapshot's
        card_uuids.

        Private helper - single consumer is decks_for().
        Inputs: row. Output: GenericDeck (name "", provenance None).
        Side effects: none. Exceptions: none.
        """
        snapshot = self._snapshot
        start = int(snapshot.deck_offsets[row])
        end = int(snapshot.deck_offsets[row + 1])

        # Expand the multiset, then name each copy by its uuid
        copies = np.repeat(
            snapshot.card_indices[start:end], snapshot.card_counts[start:end]
        )
        card_uuids = snapshot.card_snapshot.card_uuids
        return GenericDeck(
            nocab_uuid=_uuid_of(snapshot.deck_uuid_bytes[row]),
            source_game=snapshot.source_game,
            name="",
            card_nocab_uuids=[card_uuids[index] for index in copies.tolist()],
            provenance=None,
        )


def _uuid_of(uuid_bytes: np.ndarray) -> UUID:
    """A deck's uuid from its 16-byte row of deck_uuid_bytes.

    Private helper - single consumer is DeckBoxDealer._deck_at.
    Inputs: uuid_bytes (uint8, shape (16,)). Output: UUID.
    Side effects: none. Exceptions: none.
    """
    return UUID(bytes=uuid_bytes.tobytes())
