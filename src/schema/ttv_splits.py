"""Train/Test/Validate split ratios and the index math to apply them.

TTVSplits holds a list of ratios (typically length 2: train/test, or 3:
train/test/validate) and turns them into either a partition of an actual
collection (split_collection) or just the [start, end) index ranges a
collection of a given size would be cut into (get_split_indices) - the
latter is what HoldoutSpec (src/schema/holdout.py) uses to carve a fixed
hash range into tiers without ever materializing a collection.
"""

import random
from typing import Generic, Tuple, TypeVar

T = TypeVar("T")


class TTVSplits(Generic[T]):
    """Utility for helping compute Train Test and Validate splits."""

    def __init__(self, split_ratios: list[float] | None = None):
        """
        Inputs:
            split_ratios: relative sizes of each split, in order (e.g.
                [0.8, 0.1, 0.1] for an 80/10/10 train/test/validate
                split). None (the default) means no splits at all,
                equivalent to passing []. Not required to sum to 1 -
                split_collection/get_split_indices treat these as
                relative weights, not already-normalized percentages
                (see from_unnormalized for turning raw ratios like
                [8, 1, 1] into this).
        Output: none (constructor).
        Side effects: none. The given list is copied, not aliased, so a
            caller mutating its own list afterwards can't change this
            instance's split_ratios out from under it.
        Exceptions: none.

        Example:
            >>> splits = TTVSplits([0.8, 0.1, 0.1])
            >>> splits.num_splits
            3
        """
        # Typically size 2 (train, test) or 3 (train, test, validate).
        # Copied (not aliased) so a caller can't mutate this instance's
        # split_ratios via their own list after construction, and so a
        # mutable default argument can't be silently shared/mutated
        # across every call that omits split_ratios.
        self.split_ratios: list[float] = list(split_ratios) if split_ratios else []
        self.num_splits: int = len(self.split_ratios)

    @classmethod
    def from_unnormalized(cls, unnormalized_ratios: list[float]) -> "TTVSplits[T]":
        """Build a TTVSplits from ratios that don't yet sum to 1.

        Useful for specifying splits as larger number rations. For example:
        [10, 1] (or "10 to 1" train to test splits) -> ~[0.91, 0.09]
        Or "10 to 1 to 1" train to test to validate splits -> ~[0.90, 0.09, 0.01]

        Inputs:
            unnormalized_ratios: relative split sizes in any units (e.g.
                [8, 1, 1]); must sum to a positive number.
        Output: a TTVSplits whose split_ratios sum to 1 (each entry
            divided by the input's total).
        Side effects: none.
        Exceptions: ZeroDivisionError if unnormalized_ratios sums to 0
            (e.g. an empty list, or all-zero entries).

        Example:
            >>> splits = TTVSplits.from_unnormalized([8, 1, 1])
            >>> splits.get_percentages()
            [0.8, 0.1, 0.1]
        """
        total = sum(unnormalized_ratios)
        return cls([ratio / total for ratio in unnormalized_ratios])

    def split_collection(
        self,
        collection: list[T],
        shuffle: bool = False,
        rng_seed: int | None = None,
    ) -> list[list[T]]:
        """Partition collection into one sub-list per split.

        Splits a collection into a list of lists, where each list is a split.

        Inputs:
            collection: the items to partition. Not mutated in place,
                except that shuffle=True shuffles it in place (via
                random.shuffle) before splitting, same as calling
                random.shuffle on it directly.
            shuffle: if True, shuffle collection in place before
                splitting.
            rng_seed: if given (and shuffle is True), seeds the global
                random module (via random.seed) before shuffling, for a
                reproducible order. Ignored if shuffle is False.
        Output: one list[T] per entry in split_ratios, in the same
            order, sized proportionally to each ratio (rounded down to
            the nearest whole item per split); any leftover items from
            rounding go to the last split. If num_splits is 0 or 1, the
            whole (possibly shuffled) collection is returned as the
            single entry of a one-element list.
        Side effects: mutates collection in place if shuffle is True
            (and, if rng_seed is given, reseeds the global random
            module).
        Exceptions: none.

        Example:
            >>> splits = TTVSplits([0.5, 0.5])
            >>> splits.split_collection([1, 2, 3, 4])
            [[1, 2], [3, 4]]
        """
        if shuffle:
            if rng_seed is not None:
                random.seed(rng_seed)
            random.shuffle(collection)

        if self.num_splits == 0:
            return [collection]
        elif self.num_splits == 1:
            return [collection]

        result: list[list[T]] = []

        left_i: int = 0  # Represents the first index of the current split
        right_i: int = 0  # Represents the last index + 1 of the current split
        for split_idx in range(0, self.num_splits):
            left_i = right_i
            # round down to nearest integer represengting the last element in the
            # current split
            right_i = left_i + int(self.split_ratios[split_idx] * len(collection))
            split: list[T] = collection[left_i:right_i]

            result.append(split)

        # Last item(s) goes to the last split
        if right_i < len(collection):
            result[-1].extend(collection[right_i:])

        return result

    def get_percentages(self) -> list[float]:
        """Return this instance's split_ratios.

        Inputs: none.
        Output: the split_ratios list given at construction (or derived
            by from_unnormalized), in order. This is the live list, not
            a copy - a caller that mutates it also mutates this
            instance's split_ratios.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> TTVSplits([0.8, 0.1, 0.1]).get_percentages()
            [0.8, 0.1, 0.1]
        """
        return self.split_ratios

    def get_split_indices(self, collection_size: int) -> list[Tuple[int, int]]:
        """Return each split's [start, end) index range over a collection of a given size.

        Given a collection size, return the indices of the start and end of each split.
        Start index is inclusive, end index is exclusive. Unlike
        split_collection, this never materializes or touches an actual
        collection - useful when only the index boundaries are needed
        (e.g. HoldoutSpec carving a fixed hash range into tiers).

        Inputs:
            collection_size: the size of the (possibly hypothetical)
                collection to compute index ranges over.
        Output: one (start, end) tuple per entry in split_ratios, in
            order, covering [0, collection_size) with no gaps: each
            split's size is its ratio times collection_size (rounded
            down), and any leftover from rounding is folded into the
            last split's end index.
        Side effects: none.
        Exceptions: IndexError if split_ratios is empty (there is no
            last split to extend to collection_size).

        Example:
            >>> TTVSplits([0.5, 0.5]).get_split_indices(4)
            [(0, 2), (2, 4)]
        """
        result: list[Tuple[int, int]] = []
        start_index = 0
        for ratio in self.split_ratios:
            end_index = start_index + int(ratio * collection_size)
            result.append((start_index, end_index))
            start_index = end_index

        # If there's an off by one error, then the last item goes to the last split
        if result[-1][1] < collection_size:
            result[-1] = (result[-1][0], collection_size)

        return result
