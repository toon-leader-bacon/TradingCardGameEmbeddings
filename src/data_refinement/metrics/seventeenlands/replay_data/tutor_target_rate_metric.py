"""TutorTargetRateMetric - BRAINSTORM.md's "Tutor Target Rate" over replays: P(tutored at
least once in the game | card in deck_<name>).

A DeckEventRateMetric (deck_event_rate_metric.py) reading the user's
cards_tutored entries.

A distinct class from game_data's TutorTargetRateMetric
(../game_data/tutor_target_rate_metric.py): each lives in its own
source-specific package.
"""

from typing import ClassVar

from src.data_refinement.metrics.seventeenlands.replay_data.deck_event_rate_metric import (
    DeckEventRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    ReplayField,
)


class TutorTargetRateMetric(DeckEventRateMetric):
    """Card -> P(the user tutors it this game | it is in the deck)."""

    OUTPUT_STEM: ClassVar[str] = "tutor_target_rate"
    LABEL_COLUMN: ClassVar[str] = "tutor_target_rate"
    FIELDS: ClassVar[tuple[ReplayField, ...]] = (ReplayField.CARDS_TUTORED,)
