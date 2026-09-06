import random
from typing import Tuple, TypeVar, Generic

T = TypeVar("T")


class TTVSplits(Generic[T]):
    """Utility for helping compute Train Test and Validate splits."""

    def __init__(self, split_ratios: list[float] = []):
        # Typically size 2 (train, test) or 3 (train, test, validate)
        self.split_ratios: list[float] = split_ratios
        self.num_splits: int = len(split_ratios)

    @classmethod
    def from_unnormalized(cls, unnormalized_ratios: list[float]) -> "TTVSplits[T]":
        """
        Useful for specifying splits as larger number rations. For example:
        [10, 1] (or "10 to 1" train to test splits) -> ~[0.91, 0.09]
        Or "10 to 1 to 1" train to test to validate splits -> ~[0.90, 0.09, 0.01]
        """
        total = sum(unnormalized_ratios)
        return cls([ratio / total for ratio in unnormalized_ratios])

    def split_collection(
        self,
        collection: list[T],
        shuffle: bool = False,
        rng_seed: int | None = None,
    ) -> list[list[T]]:
        """
        Splits a collection into a list of lists, where each list is a split.
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
        return self.split_ratios

    def get_split_indices(self, collection_size: int) -> list[Tuple[int, int]]:
        # Given a collection size, return the indices of the start and end of each split
        # Start index is inclusive, end index is exclusive
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
