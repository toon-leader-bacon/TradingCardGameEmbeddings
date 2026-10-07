"""CastRateMetric - BRAINSTORM.md's "Cast Rate": P(cast at least once in the game |
card in deck_<name>).

A DeckEventRateMetric (deck_event_rate_metric.py) reading the user's
creatures_cast and non_creatures_cast entries.
"""

from typing import ClassVar

from src.data_refinement.metrics.seventeenlands.replay_data.deck_event_rate_metric import (
    DeckEventRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    CAST_FIELDS,
    ReplayField,
)


class CastRateMetric(DeckEventRateMetric):
    """Card -> P(the user casts it this game | it is in the deck)."""

    OUTPUT_STEM: ClassVar[str] = "cast_rate"
    LABEL_COLUMN: ClassVar[str] = "cast_rate"
    FIELDS: ClassVar[tuple[ReplayField, ...]] = CAST_FIELDS
