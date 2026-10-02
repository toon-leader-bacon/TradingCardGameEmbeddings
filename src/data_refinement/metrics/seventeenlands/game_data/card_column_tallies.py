"""CardColumnTallies - numeric tallies kept per matched header column of
one zone across a CSV's chunks, then summed per card uuid
(see game_data/README.md).

Shared by every vectorized per-card game_data metric:
GameCardAverageMetric (float tallies: a value sum and a count),
OnPlayWinRateDeltaMetric and TutorTargetRateMetric (integer counts).
Tallying per column and grouping by card only at the end keeps the row
implementation's counting exactly: two header columns naming one card
both count, as the row implementation's present_uuids() listed that
card twice.
"""

from typing import Generic, TypeVar
from uuid import UUID

import numpy as np
import numpy.typing as npt

TallyT = TypeVar("TallyT", np.int64, np.float64)


class CardColumnTallies(Generic[TallyT]):
    """tally_count tallies of one dtype per column of one zone's
    layout, the layout fixed by the first chunk."""

    def __init__(self, owner: str, tally_count: int, dtype: type[TallyT]) -> None:
        """Start with no layout and no tallies; the first add() sets
        both.

        Inputs:
            owner: the metric's name, for error messages.
            tally_count: how many tallies each column keeps (e.g. 2 for
                (times in deck, times tutored)).
            dtype: the tallies' dtype, np.int64 or np.float64.
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if tally_count < 1.
        """
        if tally_count < 1:
            raise ValueError(f"{owner}: tally_count must be >= 1")
        self._owner = owner
        self._tally_count = tally_count
        self._dtype: type[TallyT] = dtype
        self._card_uuids: tuple[UUID, ...] | None = None
        self._tallies: npt.NDArray[TallyT] | None = None

    def add(self, card_uuids: tuple[UUID, ...], increments: npt.NDArray) -> None:
        """Add one chunk's per-column increments.

        Inputs:
            card_uuids: the zone's columns (ZoneCounts.card_uuids).
            increments: shape (tally_count, len(card_uuids)), castable
                to dtype.
        Output: none.
        Side effects: sets the layout on the first call; adds
            increments to the tallies.
        Exceptions: ValueError if card_uuids differs from the first
            call's (chunks from two CSVs), increments' shape is wrong,
            or float increments meet integer tallies (which would
            truncate).

        Example:
            >>> tallies = CardColumnTallies("TutorTargetRateMetric", 2, np.int64)
            >>> tallies.add(deck.card_uuids, np.stack([in_deck, tutored]))
        """
        # Validate inputs: shape, then the same layout as every earlier call
        expected_shape = (self._tally_count, len(card_uuids))
        if increments.shape != expected_shape:
            raise ValueError(
                f"{self._owner}: increments shape {increments.shape}, "
                f"expected {expected_shape}"
            )
        if increments.dtype.kind == "f" and np.dtype(self._dtype).kind != "f":
            raise ValueError(f"{self._owner}: float increments for integer tallies")
        tallies = self._tallies_for(card_uuids)

        tallies += increments.astype(self._dtype, copy=False)

    def per_card(self) -> dict[UUID, npt.NDArray[TallyT]]:
        """The tallies summed per card uuid, in first-column order.

        Inputs: none.
        Output: dict card uuid -> shape (tally_count,) tallies, for
            every card with a column (all-zero cards included); empty
            if add() was never called.
        Side effects: none. Exceptions: none.

        Example:
            >>> in_deck, tutored = tallies.per_card()[owlbear_uuid]
        """
        result: dict[UUID, npt.NDArray[TallyT]] = {}
        if self._card_uuids is None or self._tallies is None:
            return result

        # Columns naming the same card add up
        for column, card_uuid in enumerate(self._card_uuids):
            column_tallies = self._tallies[:, column]
            previous = result.get(card_uuid)
            result[card_uuid] = (
                column_tallies.copy() if previous is None else previous + column_tallies
            )
        return result

    def _tallies_for(self, card_uuids: tuple[UUID, ...]) -> npt.NDArray[TallyT]:
        """The running tallies, after recording the first layout (and
        zeroing them) or checking a later one against it.

        Inputs: card_uuids. Output: the tallies array (mutable, owned
            by this object).
        Side effects: sets the layout and zero tallies on the first
            call.
        Exceptions: ValueError on a layout mismatch.
        """
        if self._card_uuids is None or self._tallies is None:
            self._card_uuids = card_uuids
            self._tallies = np.zeros((self._tally_count, len(card_uuids)), self._dtype)
            return self._tallies
        if card_uuids != self._card_uuids:
            raise ValueError(
                f"{self._owner}: chunk's columns differ from the first chunk's "
                "(chunks from two CSVs?)"
            )
        return self._tallies
