"""DiscardRateMetric - BRAINSTORM.md's "Discard Rate": P(discarded at least once in the
game | card in deck_<name>).

A DeckEventRateMetric (deck_event_rate_metric.py) reading the user's
cards_discarded entries.

Unverified assumption: every user_turn_N_cards_discarded entry is
self-inflicted, never a forced discard from an opposing effect.
"""

from typing import ClassVar

from src.data_refinement.metrics.seventeenlands.replay_data.deck_event_rate_metric import (
    DeckEventRateMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    ReplayField,
)


class DiscardRateMetric(DeckEventRateMetric):
    """Card -> P(the user discards it this game | it is in the deck)."""

    OUTPUT_STEM: ClassVar[str] = "discard_rate"
    LABEL_COLUMN: ClassVar[str] = "discard_rate"
    FIELDS: ClassVar[tuple[ReplayField, ...]] = (ReplayField.CARDS_DISCARDED,)
