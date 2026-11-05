"""NeighbourRule: how many of a card's most similar teacher rows count
as its neighbours. A Strategy: find_neighbours (teacher_neighbours.py)
ranks every card's candidates once, by cosine, and asks the rule how
many of them to keep.

Every rule's contract: kept_count receives one card's candidates
ranked cosine-descending, possibly ending in -inf padding (a card with
fewer candidates than candidate_limit), and never keeps a non-finite
cosine.

A new rule is one more frozen dataclass with KIND, candidate_limit,
kept_count and payload, added to the NeighbourRule union and to the
config parser's table.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class TopKNeighbours:
    """Each card's k most similar rows, however far they are."""

    KIND: ClassVar[str] = "top_k"

    k: int

    def __post_init__(self) -> None:
        """Raise ValueError for k <= 0."""
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")

    @property
    def candidate_limit(self) -> int:
        """At most this many candidates per card are ranked.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return self.k

    def kept_count(self, ranked_cosines: np.ndarray) -> int:
        """How many of one card's ranked candidates to keep: the first k
        finite ones.

        Inputs: ranked_cosines (np.ndarray float32, descending, -inf
            padded; see the module docstring).
        Output: int, <= the finite count.
        Side effects: none. Exceptions: none.

        Example:
            >>> TopKNeighbours(2).kept_count(np.array([0.9, -np.inf, -np.inf]))
            1
        """
        return min(self.k, int(np.isfinite(ranked_cosines).sum()))

    def payload(self) -> dict[str, object]:
        """The rule's kind and options, JSON-ready (the neighbours file's
        name and stamp).

        Inputs: none. Output: dict. Side effects: none. Exceptions: none.
        """
        return {"kind": self.KIND, "k": self.k}


@dataclass(frozen=True)
class CosineThresholdNeighbours:
    """Every row at least min_cosine similar, at most max_neighbours of
    them (the most similar)."""

    KIND: ClassVar[str] = "cosine_threshold"

    min_cosine: float
    max_neighbours: int

    def __post_init__(self) -> None:
        """Raise ValueError for min_cosine outside [-1, 1] or max_neighbours <= 0."""
        if not -1.0 <= self.min_cosine <= 1.0:
            raise ValueError(f"min_cosine must be in [-1, 1], got {self.min_cosine}")
        if self.max_neighbours <= 0:
            raise ValueError(
                f"max_neighbours must be positive, got {self.max_neighbours}"
            )

    @property
    def candidate_limit(self) -> int:
        """At most this many candidates per card are ranked.

        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        return self.max_neighbours

    def kept_count(self, ranked_cosines: np.ndarray) -> int:
        """How many of one card's ranked candidates reach min_cosine.

        Inputs: ranked_cosines (np.ndarray float32, descending, -inf
            padded; see the module docstring). Output: int.
        Side effects: none. Exceptions: none.

        Example:
            >>> CosineThresholdNeighbours(0.4, 8).kept_count(np.array([0.9, 0.5, 0.1]))
            2
        """
        # -inf padding is below every min_cosine in [-1, 1]
        reached = int(np.count_nonzero(ranked_cosines >= self.min_cosine))
        return min(self.max_neighbours, reached)

    def payload(self) -> dict[str, object]:
        """The rule's kind and options, JSON-ready (the neighbours file's
        name and stamp).

        Inputs: none. Output: dict. Side effects: none. Exceptions: none.
        """
        return {
            "kind": self.KIND,
            "min_cosine": self.min_cosine,
            "max_neighbours": self.max_neighbours,
        }


# Every neighbour rule; a new rule adds its class here
NeighbourRule = TopKNeighbours | CosineThresholdNeighbours
