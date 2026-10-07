"""Concrete PackCardTallyMetric (pack_card_tally_metric.py) subclasses:
the three take rates.

Each sets only OUTPUT_STEM and KEY_COLUMNS (its stratification), plus,
where needed, which rows are eligible.
"""

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk import (
    DraftDataChunk,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metric import (
    PackCardTallyMetric,
)


class CardTakeRateMetric(PackCardTallyMetric):
    """P(picked | in pack, pick_number, pack_number) - BRAINSTORM.md's
    single-card metric #1 (Card Take Rate)."""

    OUTPUT_STEM: ClassVar[str] = "card_take_rate"
    KEY_COLUMNS: ClassVar[tuple[str, ...]] = (
        "nocab_uuid",
        "pack_number",
        "pick_number",
    )


class FirstPickRateMetric(PackCardTallyMetric):
    """P(picked | pack_number == 0, pick_number == 0, in pack) -
    BRAINSTORM.md's single-card metric #4 (First-Pick Rate).

    Keyed by the card alone: the eligibility restriction to pack 0 /
    pick 0 does all of the conditioning.
    """

    OUTPUT_STEM: ClassVar[str] = "first_pick_rate"
    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ("nocab_uuid",)

    def _eligible_rows(self, chunk: DraftDataChunk) -> npt.NDArray[np.bool_]:
        """See PackCardTallyMetric._eligible_rows(): pack 0, pick 0 only.

        Inputs: chunk. Output: bool array (rows,).
        Side effects: none. Exceptions: none.
        """
        return (chunk.pack_number == 0) & (chunk.pick_number == 0)


class RankStratifiedTakeRateMetric(PackCardTallyMetric):
    """Card Take Rate, also stratified by rank - BRAINSTORM.md's
    single-card metric #6 (Rank-Stratified Take Rate)."""

    OUTPUT_STEM: ClassVar[str] = "rank_stratified_take_rate"
    KEY_COLUMNS: ClassVar[tuple[str, ...]] = (
        "nocab_uuid",
        "pack_number",
        "pick_number",
        "rank",
    )

    def _eligible_rows(self, chunk: DraftDataChunk) -> npt.NDArray[np.bool_]:
        """See PackCardTallyMetric._eligible_rows(): ranked rows only (a
        Trad/Sealed row's rank is "", which has no stratum).

        Inputs: chunk. Output: bool array (rows,).
        Side effects: none. Exceptions: none.
        """
        return chunk.rank != ""
