"""PoolConditionedPickMetric - BRAINSTORM.md's multi-group metric #5
(Pool-Conditioned Pick Prediction): given the pool so far AND the
current pack options, which option gets taken.

A DraftChoiceStreamMetric (draft_choice_stream_metric.py) with
INCLUDES_POOL: one extra list column, pool_uuids, the drafter's pool so
far, read as-is from the pool_<name> columns (they already exclude this
row's own pick). A card held twice is listed once, as the row
implementation did.
"""

from typing import ClassVar

from src.data_refinement.metrics.seventeenlands.draft_data.draft_choice_stream_metric import (
    DraftChoiceStreamMetric,
)


class PoolConditionedPickMetric(DraftChoiceStreamMetric):
    """(pool so far, pack options) -> which option was taken."""

    OUTPUT_STEM: ClassVar[str] = "pool_conditioned_pick"
    INCLUDES_POOL: ClassVar[bool] = True
