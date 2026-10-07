"""TutorChoiceRateMetric - when a tutor was used, how often it fetched
this card: per card, P(card tutored | card in deck_<name>, at least one
card tutored that game).

Replaces TutorTargetPoolMetric, which streamed one row per (game, card
in deck or sideboard) with a tutored bool (about 1.2B rows over the full
corpus). Only the per-card rate is a training signal, so this is a count
table. Deck only: a sideboard card is not counted.

A TutorTargetRateMetric (tutor_target_rate_metric.py) whose
_counted_rows() keeps only games with a tutored card: the same counts
over a smaller denominator, so the rate reads as "how often a tutor
picks this card" rather than "how often this card gets tutored at all".
"""

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from src.data_refinement.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)


class TutorChoiceRateMetric(TutorTargetRateMetric):
    """Card -> P(tutored | in deck, at least one card tutored that game)."""

    OUTPUT_STEM: ClassVar[str] = "tutor_choice_rate"
    LABEL_COLUMN: ClassVar[str] = "tutor_choice_rate"

    def _counted_rows(self, chunk: GameDataChunk) -> npt.NDArray[np.bool_]:
        """Only games where some card was tutored count.

        Inputs: chunk.
        Output: shape (len(chunk),); all False when the chunk has no
            tutored column.
        Side effects: none. Exceptions: none.
        """
        return chunk.zones[GameZone.TUTORED].present().any(axis=1)
