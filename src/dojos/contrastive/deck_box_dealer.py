"""Mechanical train/test/validation wrapper over a DeckBox.

See plans/contrastive_dojo.md's "DeckBoxDealer" section. Parallel in
role to FileManagerParquet, but simpler: DeckBox is already fully in
memory, so there is no disk-streaming step. Partitions one game's deck
uuids into train/test/validation groups once, at construction (seeded),
and yields groups of raw, unmodified GenericDeck objects per split - a
"deck sample," deliberately not "batch" (that word is reserved for
ContrastiveBatch, built later by a ContrastivePairConstructor). This
class has no concept of "anchor" or "negative" deck - that framing
belongs entirely to ContrastivePairConstructor.
"""

import random
from typing import Iterator
from uuid import UUID

from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.splits import Split


class DeckBoxDealer:
    """Owns one game's train/test/validation uuid partition and deals
    out raw deck samples from it. Non-swappable - the plan treats this
    as mechanical plumbing, not a research surface."""

    def __init__(
        self,
        deck_box: DeckBox,
        source_game: GameId,
        split_ratios: list[float] = [8, 1, 1],
        seed: int | None = None,
    ) -> None:
        """Partition source_game's decks into train/test/validation groups.

        Inputs:
            deck_box: the fully-loaded deck store to deal from.
            source_game: which game's decks to partition. Only this
                game's uuids (deck_box.all_uuids(source_game)) are
                considered.
            split_ratios: relative train/test/validation sizes, in that
                order (e.g. [8, 1, 1] is an 80/10/10 split). Need not
                sum to any particular total - only the ratio matters.
            seed: seed for the partition shuffle. None means
                non-deterministic.
        Output: none (constructor).
        Side effects: none beyond reading deck_box.all_uuids().
        Exceptions: ValueError if split_ratios doesn't have exactly 3
            entries, or any entry is negative.

        Example:
            >>> dealer = DeckBoxDealer(deck_box, GameId.MTG, seed=42)
        """
        if len(split_ratios) != 3:
            raise ValueError(
                f"split_ratios must have exactly 3 entries, got {len(split_ratios)}"
            )
        if any(ratio < 0 for ratio in split_ratios):
            raise ValueError("split_ratios entries must be non-negative")
        if sum(split_ratios) <= 0:
            raise ValueError("split_ratios must sum to a positive value")

        self._deck_box = deck_box
        self._source_game = source_game
        self._rng = random.Random(seed)

        all_uuids = list(deck_box.all_uuids(source_game))
        self._rng.shuffle(all_uuids)
        self._train_uuids, self._test_uuids, self._validation_uuids = (
            self._partition_uuids(all_uuids, split_ratios)
        )

    def decks_for(
        self, split: Split, decks_per_sample: int, shuffle: bool = True
    ) -> Iterator[list[GenericDeck]]:
        """Yield any split's decks, grouped into samples.

        Inputs: split (Split), plus training_decks()'s decks_per_sample
            and shuffle.
        Output: a generator of lists of GenericDeck, each of length
            decks_per_sample.
        Side effects: none beyond reading from self._deck_box.
        Exceptions: ValueError if decks_per_sample <= 0.
        """
        yield from self._decks_for_split(
            self._uuids_of(split), decks_per_sample, shuffle
        )

    def deck_count(self, split: Split) -> int:
        """Number of decks partitioned into a split.

        Inputs: split (Split). Output: int. Side effects: none.
        Exceptions: none.
        """
        return len(self._uuids_of(split))

    def _uuids_of(self, split: Split) -> list[UUID]:
        return {
            Split.TRAIN: self._train_uuids,
            Split.TEST: self._test_uuids,
            Split.VALIDATION: self._validation_uuids,
        }[split]

    def training_decks(
        self, decks_per_sample: int, shuffle: bool = True
    ) -> Iterator[list[GenericDeck]]:
        """Yield the training split's decks, grouped into samples.

        Inputs:
            decks_per_sample: how many decks per yielded group. Any
                remainder that doesn't fill a full group is dropped, so
                every yielded group has exactly this many decks.
            shuffle: whether to reshuffle this split's uuid order before
                grouping (e.g. once per epoch). True by default.
        Output: a generator of lists of GenericDeck, each of length
            decks_per_sample.
        Side effects: none beyond reading from self._deck_box.
        Exceptions: ValueError if decks_per_sample <= 0.

        Example:
            >>> for deck_sample in dealer.training_decks(decks_per_sample=3):
            ...     ...
        """
        yield from self._decks_for_split(self._train_uuids, decks_per_sample, shuffle)

    def test_decks(
        self, decks_per_sample: int, shuffle: bool = True
    ) -> Iterator[list[GenericDeck]]:
        """Yield the test split's decks, grouped into samples.

        Inputs: see training_decks() - identical shape, sourced from
            the test partition instead.
        Output: a generator of lists of GenericDeck, each of length
            decks_per_sample.
        Side effects: none beyond reading from self._deck_box.
        Exceptions: ValueError if decks_per_sample <= 0.

        Example:
            >>> for deck_sample in dealer.test_decks(decks_per_sample=3):
            ...     ...
        """
        yield from self._decks_for_split(self._test_uuids, decks_per_sample, shuffle)

    def validation_decks(
        self, decks_per_sample: int, shuffle: bool = True
    ) -> Iterator[list[GenericDeck]]:
        """Yield the validation split's decks, grouped into samples.

        Inputs: see training_decks() - identical shape, sourced from
            the validation partition instead.
        Output: a generator of lists of GenericDeck, each of length
            decks_per_sample.
        Side effects: none beyond reading from self._deck_box.
        Exceptions: ValueError if decks_per_sample <= 0.

        Example:
            >>> for deck_sample in dealer.validation_decks(decks_per_sample=3):
            ...     ...
        """
        yield from self._decks_for_split(
            self._validation_uuids, decks_per_sample, shuffle
        )

    def _partition_uuids(
        self, uuids: list[UUID], split_ratios: list[float]
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        """Split an already-shuffled uuid list into 3 groups by ratio.

        Private helper - single caller is __init__.

        Inputs:
            uuids: already-shuffled uuids to split.
            split_ratios: see __init__.
        Output: (train_uuids, test_uuids, validation_uuids).
        Side effects: none.
        Exceptions: none.
        """
        total_ratio = sum(split_ratios)
        item_count = len(uuids)
        train_end = round(item_count * split_ratios[0] / total_ratio)
        test_end = round(item_count * (split_ratios[0] + split_ratios[1]) / total_ratio)
        return uuids[:train_end], uuids[train_end:test_end], uuids[test_end:]

    def _decks_for_split(
        self, uuids: list[UUID], decks_per_sample: int, shuffle: bool
    ) -> Iterator[list[GenericDeck]]:
        """Shared uuid-group-to-GenericDeck-list conversion for
        training_decks/test_decks/validation_decks.

        Private helper - single set of callers are the three public
        split methods above. Mirrors the existing generic dojo cells'
        `_data_iterator` convention (e.g.
        SingleCardRegressionDojo._data_iterator).

        Inputs:
            uuids: this split's uuid list.
            decks_per_sample: see training_decks().
            shuffle: whether to reshuffle uuids (a fresh copy - this
                split's own stored order is never mutated) before
                grouping.
        Output: a generator of lists of GenericDeck, each of length
            decks_per_sample. A trailing partial group is dropped.
        Side effects: none beyond reading from self._deck_box.
        Exceptions: ValueError if decks_per_sample <= 0.
        """
        if decks_per_sample <= 0:
            raise ValueError("decks_per_sample must be positive")

        # Work on a copy so this split's stored uuid order is untouched.
        ordered_uuids = list(uuids)
        if shuffle:
            self._rng.shuffle(ordered_uuids)

        # Walk the (possibly reshuffled) uuids in fixed-size groups,
        # dropping a trailing group that can't be filled completely.
        for start in range(0, len(ordered_uuids), decks_per_sample):
            group_uuids = ordered_uuids[start : start + decks_per_sample]
            if len(group_uuids) < decks_per_sample:
                break
            yield self._decks_by_uuids(group_uuids)

    def _decks_by_uuids(self, uuids: list[UUID]) -> list[GenericDeck]:
        """Look up a group of decks by uuid, in order.

        Private helper - single caller is _decks_for_split.

        Inputs:
            uuids: uuids to look up, all expected to be present (they
                came from this dealer's own partition of
                deck_box.all_uuids()).
        Output: the matching GenericDeck objects, same order as uuids.
        Side effects: none.
        Exceptions: none expected (see Inputs) - would raise
            RuntimeError if self._deck_box is mutated out from under
            this dealer after construction.
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
