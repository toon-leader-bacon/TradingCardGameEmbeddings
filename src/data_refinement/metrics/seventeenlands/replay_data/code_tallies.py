"""CodeTallies - float64 tallies per card code, the per-card state of the
replay_data event metrics (see replay_data/README.md).

A replay chunk names cards by code (ReplayDataChunk.card_uuids is the
code table), and codes are stable for a CSV, so each tally is one numpy
array indexed by code, grown as new codes appear. Per chunk, add() is one
np.bincount per tally.
"""

from collections.abc import Callable
from uuid import UUID

import numpy as np
import numpy.typing as npt


class CodeTallies:
    """tally_count float64 tallies per card code, summed over a CSV.

    float64 although counts are whole numbers: it matches every other
    17lands count table's count dtype, and sums (turns) share the array.
    """

    def __init__(self, owner: str, tally_count: int) -> None:
        """Start with no codes.

        Inputs: owner (named in errors), tally_count (tallies per code).
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if tally_count < 1.
        """
        if tally_count < 1:
            raise ValueError(f"{owner}: tally_count must be >= 1")
        self._owner = owner
        self._tally_count = tally_count
        self._tallies = np.zeros((tally_count, 0), np.float64)

    def add(
        self,
        tally: int,
        codes: npt.NDArray[np.int32],
        weights: npt.NDArray[np.generic] | None = None,
    ) -> None:
        """Add one entry per code to one tally: 1 each, or weights.

        Inputs:
            tally: which tally (0 .. tally_count - 1).
            codes: card codes (>= 0), one per entry; may repeat.
            weights: one value per entry (e.g. the turn), or None for 1.
        Output: none.
        Side effects: grows the code axis to fit; adds to the tally.
        Exceptions: ValueError if tally is out of range, a code is
            negative, or weights' length differs from codes'.

        Example:
            >>> tallies.add(0, cast.codes)               # occurrences
            >>> tallies.add(1, cast.codes, cast.turns)   # turn sums
        """
        if not 0 <= tally < self._tally_count:
            raise ValueError(f"{self._owner}: no tally {tally}")
        if weights is not None and weights.shape != codes.shape:
            raise ValueError(
                f"{self._owner}: {weights.shape[0]} weights for {codes.shape[0]} codes"
            )
        if codes.size == 0:
            return
        if codes.min() < 0:
            raise ValueError(f"{self._owner}: negative card code")

        # Grow the code axis to fit, then add one bincount
        code_count = int(codes.max()) + 1
        if code_count > self._tallies.shape[1]:
            grown = np.zeros((self._tally_count, code_count), np.float64)
            grown[:, : self._tallies.shape[1]] = self._tallies
            self._tallies = grown
        size = self._tallies.shape[1]
        if weights is None:
            self._tallies[tally] += np.bincount(codes, minlength=size)
        else:
            self._tallies[tally] += np.bincount(
                codes, weights=weights.astype(np.float64), minlength=size
            )

    def count_columns(
        self,
        card_uuids: tuple[UUID, ...],
        names: tuple[str, ...],
        keep: Callable[[npt.NDArray[np.float64]], bool],
    ) -> tuple[list[str], dict[str, npt.NDArray[np.float64]]]:
        """The kept codes as count-table columns, one row per card.

        Inputs:
            card_uuids: the code table (card_uuids[code]); covers every
                code tallied.
            names: one column name per tally.
            keep: whether a code's tallies (shape (tally_count,)) earn a
                row (e.g. at least one occurrence).
        Output: (card uuid strings, {name: float64 array}), in code
            order. Two codes never name one card (the code table is one
            code per card).
        Side effects: none.
        Exceptions: ValueError if names has not one name per tally, or
            card_uuids is shorter than the tallied codes.

        Example:
            >>> tallies.count_columns(chunk.card_uuids, ("count", "turn_sum"), keep)
        """
        if len(names) != self._tally_count:
            raise ValueError(
                f"{self._owner}: {len(names)} names for {self._tally_count} tallies"
            )
        code_count = self._tallies.shape[1]
        if len(card_uuids) < code_count:
            raise ValueError(
                f"{self._owner}: {len(card_uuids)} card uuids for {code_count} codes"
            )

        kept = [code for code in range(code_count) if keep(self._tallies[:, code])]
        counts = {name: self._tallies[i, kept] for i, name in enumerate(names)}
        return [str(card_uuids[code]) for code in kept], counts
