

import math
import random
from typing import Generic, TypeVar


T = TypeVar('T')


class SplitSamplerWithReplacement(Generic[T]):
    def __init__(self,
                 collection: list[T],
                 rng_seed: int | None = None):
        self.collection = collection
        self.rng = random.Random(rng_seed) \
            if rng_seed is not None \
            else random.Random()

    def sample(self, batch_size: int, seed: int | None = None) -> list[T]:
        if batch_size <= 0 or len(self.collection) == 0:
            return []
        if seed is not None:
            self.rng = random.Random(seed)
        return [self.rng.choice(self.collection) for _ in range(batch_size)]


class SplitSamplerWellOrdered(Generic[T]):
    def __init__(self,
                 collection: list[T],
                 rng_seed: int | None = None):
        self.collection = collection
        self.next_idx = 0
        self.length = len(collection)

        self.rng = random.Random(rng_seed) \
            if rng_seed is not None \
            else random.Random()

    def reset(self, shuffle: bool = False, seed: int | None = None):
        self.next_idx = 0
        if seed is not None:
            self.rng = random.Random(seed)
        if shuffle:
            self.rng.shuffle(self.collection)

    def sample(self, batch_size: int) -> list[T]:
        if batch_size <= 0 or self.length == 0:
            return []
        if self.is_done():
            return []
        first_idx = self.next_idx
        last_idx = min(first_idx + batch_size, self.length)
        result = self.collection[first_idx:last_idx]
        self.next_idx = last_idx
        return result

    def is_done(self) -> bool:
        return self.next_idx >= self.length


class SplitSamplerWellOrderedLooping(Generic[T]):
    def __init__(self,
                 collection: list[T],
                 rng_seed: int | None = None):
        self.collection = collection
        self.next_idx = 0
        self.length = len(collection)
        self.rng = random.Random(rng_seed) \
            if rng_seed is not None \
            else random.Random()

    def reset(self, shuffle: bool = False, seed: int | None = None):
        self.next_idx = 0
        if seed is not None:
            self.rng = random.Random(seed)
        if shuffle:
            self.rng.shuffle(self.collection)

    def is_done(self) -> bool:
        return self.next_idx >= self.length

    def sample(self, batch_size: int) -> list[T]:
        """Sample `batch_size` elements, reshuffling at each epoch boundary."""
        if batch_size <= 0 or self.length == 0:
            return []
        result: list[T] = []
        while len(result) < batch_size:
            if self.is_done():
                self.reset(shuffle=True)
            # Take either the remaining elements needed to reach `batch_size`,
            # or the remaining elements in the collection, whichever is less.
            take = min(batch_size - len(result), self.length - self.next_idx)
            result.extend(self.collection[self.next_idx:self.next_idx + take])
            self.next_idx += take
        return result
