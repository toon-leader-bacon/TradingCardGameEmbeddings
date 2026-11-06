"""SpreadOrder: a fixed permutation of the ids 1..n whose every prefix
is near-uniform over 1..n, the "every id" half of SitemapThenSpreadPlan.
"""

import math
from dataclasses import dataclass
from typing import Any

from src.data_retrieval.download_utils import require_int

_GOLDEN_RATIO = (1 + math.sqrt(5)) / 2


@dataclass(frozen=True)
class SpreadOrder:
    """A fixed permutation of the ids 1..n: position k -> id
    ((offset + k * stride) mod n) + 1. stride is coprime to n, so every
    id appears exactly once. stride and offset are persisted in the plan
    (not recomputed from n), so a later change to for_range() can't
    reorder a running crawl.

    Invariants (checked in __post_init__): n >= 1, stride >= 1,
    gcd(stride, n) == 1, 0 <= offset < n.
    """

    n: int
    stride: int
    offset: int

    def __post_init__(self) -> None:
        """Inputs: the fields. Output: none. Side effects: none.
        Exceptions: ValueError if an invariant fails."""
        if self.n < 1:
            raise ValueError(f"spread n must be >= 1, got {self.n}")
        if self.stride < 1 or math.gcd(self.stride, self.n) != 1:
            raise ValueError(
                f"spread stride {self.stride} must be >= 1 and coprime to {self.n}"
            )
        if not 0 <= self.offset < self.n:
            raise ValueError(f"spread offset {self.offset} not in [0, {self.n})")

    @classmethod
    def for_range(cls, n: int) -> "SpreadOrder":
        """The spread order for 1..n: stride is the first coprime of n at
        or above n / golden ratio (consecutive positions land far apart;
        each prefix is near-uniform over 1..n), offset 0.

        Inputs: n (>= 1). Output: SpreadOrder.
        Side effects: none. Exceptions: ValueError if n < 1.

        Example:
            >>> SpreadOrder.for_range(10)
            SpreadOrder(n=10, stride=7, offset=0)
        """
        if n < 1:
            raise ValueError(f"spread n must be >= 1, got {n}")
        stride = max(1, round(n / _GOLDEN_RATIO))
        while math.gcd(stride, n) != 1:
            stride += 1
        return cls(n, stride, 0)

    def id_at(self, position: int) -> int:
        """The id at position (0 <= position < n).

        Inputs: position. Output: int in [1, n]. Side effects: none.
        Exceptions: IndexError outside [0, n).

        Example:
            >>> SpreadOrder(10, 7, 0).id_at(1)
            8
        """
        if not 0 <= position < self.n:
            raise IndexError(f"spread position {position} not in [0, {self.n})")
        return (self.offset + position * self.stride) % self.n + 1

    def to_dict(self) -> dict[str, int]:
        """The order as plan JSON: {"n", "stride", "offset"}, in that order.
        Inputs: none. Output: dict. Side effects: none. Exceptions: none.

        Example:
            >>> SpreadOrder(10, 7, 0).to_dict()
            {'n': 10, 'stride': 7, 'offset': 0}
        """
        return {"n": self.n, "stride": self.stride, "offset": self.offset}

    @classmethod
    def from_dict(cls, raw: Any) -> "SpreadOrder":
        """Parse to_dict()'s form (parsed JSON).
        Inputs: raw. Output: SpreadOrder. Side effects: none.
        Exceptions: KeyError, TypeError for a malformed value (not an
            object, a missing or non-int field); ValueError for a broken
            invariant.

        Example:
            >>> SpreadOrder.from_dict({"n": 10, "stride": 7, "offset": 0})
            SpreadOrder(n=10, stride=7, offset=0)
        """
        return cls(
            require_int(raw["n"]),
            require_int(raw["stride"]),
            require_int(raw["offset"]),
        )
