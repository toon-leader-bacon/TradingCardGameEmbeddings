"""PackToPickChoiceSetMetric - BRAINSTORM.md's multi-card metric #4
(Pack-to-Pick Choice Set): for every single pick, the full pack option
set plus which one was taken - a direct "options vs. choice" example.

A DraftChoiceStreamMetric (draft_choice_stream_metric.py) with no extra
columns.
"""

from typing import ClassVar

from src.data_refinement.metrics.seventeenlands.draft_data.draft_choice_stream_metric import (
    DraftChoiceStreamMetric,
)


class PackToPickChoiceSetMetric(DraftChoiceStreamMetric):
    """One pick's full pack option set -> which option was taken."""

    OUTPUT_STEM: ClassVar[str] = "pack_to_pick_choice_set"
