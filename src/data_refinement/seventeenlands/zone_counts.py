"""ZoneCounts - one card-column family's per-row copy counts for a
17lands chunk: game_data's zones (deck_, tutored_, ...) and draft_data's
pack_card_ and pool_ columns.

One matrix column per matched header column, so two header columns that
share a card uuid stay two columns, which preserves the row
implementations' per-column counting exactly.
"""

from dataclasses import dataclass
from functools import cached_property
from uuid import UUID

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ZoneCounts:
    """One zone's per-game copy counts for a chunk's rows.

    card_uuids: one card uuid per matched header column, in header
        order; may repeat (two columns matching one card).
    counts: shape (rows, len(card_uuids)); a null CSV cell is 0, which
        matches the row implementation's "NaN means absent".
    """

    card_uuids: tuple[UUID, ...]
    counts: npt.NDArray[np.int16]

    def present(self) -> npt.NDArray[np.bool_]:
        """Which (row, column) cells hold at least one copy. (The row
        implementation tested "nonzero"; a count is never negative, so
        the two agree.)

        Inputs: none. Output: bool array shaped like counts.
        Side effects: none. Exceptions: none.

        Example:
            >>> chunk.zones[GameZone.DECK].present().sum(axis=0)
        """
        return self._present_cells

    @cached_property
    def _present_cells(self) -> npt.NDArray[np.bool_]:
        """counts > 0, computed once per ZoneCounts: every metric of a
        scan reads the same chunk's zones.

        Inputs: none. Output: bool array shaped like counts.
        Side effects: caches the array on this instance, read-only (it is
            shared by every metric). Exceptions: none.
        """
        result = self.counts > 0
        result.flags.writeable = False
        return result

    def present_for(self, card_uuids: tuple[UUID, ...]) -> npt.NDArray[np.bool_]:
        """Which rows hold each of card_uuids in this zone, under any of
        its columns. Lines this zone up against another zone's columns
        (e.g. "was this deck column's card tutored?").

        Inputs: card_uuids (any cards; may repeat, may include cards
            this zone has no column for).
        Output: bool array, shape (rows, len(card_uuids)); a card with
            no column in this zone is never present.
        Side effects: none. Exceptions: none.

        Example:
            >>> deck = chunk.zones[GameZone.DECK]
            >>> chunk.zones[GameZone.TUTORED].present_for(deck.card_uuids)
        """
        rows = self.counts.shape[0]
        result = np.zeros((rows, len(card_uuids)), dtype=np.bool_)

        # One "any column of this card" vector per distinct card here
        present_per_card = self._present_per_distinct_card()

        # Copy each requested card's vector into its output column
        for output_column, card_uuid in enumerate(card_uuids):
            card_present = present_per_card.get(card_uuid)
            if card_present is not None:
                result[:, output_column] = card_present
        return result

    def _present_per_distinct_card(self) -> dict[UUID, npt.NDArray[np.bool_]]:
        """Per distinct card uuid, whether any of its columns is present,
        per row.

        Inputs: none. Output: dict card uuid -> bool array (rows,).
        Side effects: none. Exceptions: none.
        """
        present = self.present()
        result: dict[UUID, npt.NDArray[np.bool_]] = {}

        # A card's vector is the OR of its columns (usually just one)
        for column, card_uuid in enumerate(self.card_uuids):
            previous = result.get(card_uuid)
            column_present = present[:, column]
            result[card_uuid] = (
                column_present if previous is None else previous | column_present
            )
        return result
