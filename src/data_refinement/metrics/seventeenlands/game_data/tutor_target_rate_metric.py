"""TutorTargetRateMetric - BRAINSTORM.md's single-card metric "Tutor
Target Rate": P(card in tutored_<name> | card in deck_<name>) - among
games where a card was in the deck, how often did a tutor effect
actually fetch it that game.

A CardCountTableMetric (card_count_table_metric.py). Per deck column it
counts (games in deck, games in deck and tutored); "tutored" is the deck
column's card present under any tutored_<name> column, via
ZoneCounts.present_for(). The rate is taken only when a slice is built.

_counted_rows(chunk) picks the games that count (every game here).
TutorChoiceRateMetric (tutor_choice_rate_metric.py) narrows it to games
with a tutor and is otherwise this class.

Structurally close to draft_data's take-rate shape (a ratio of two
per-card tallies) but NOT a subclass of it: that class is specific to
draft_data's pack/pick concept, and reaching into draft_data/ for a base
would be the cross-cousin-directory import PRINCIPLES.md flags.

tutored_<name> is rare (~8.5% of games have any hit at all per
BRAINSTORM.md) - sample_count is written alongside the rate so a
consumer can filter out noisy near-zero estimates from cards seen in
few games.
"""

from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.count_table import ratio_output
from src.data_refinement.metrics.seventeenlands.game_data.card_count_table_metric import (
    CardCountTableMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)

IN_DECK_COLUMN = "in_deck"
TUTORED_COLUMN = "tutored"


class TutorTargetRateMetric(CardCountTableMetric):
    """Card -> P(tutored | in deck)."""

    OUTPUT_STEM: ClassVar[str] = "tutor_target_rate"
    LABEL_COLUMN: ClassVar[str] = "tutor_target_rate"
    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = (IN_DECK_COLUMN, TUTORED_COLUMN)
    ZONE: ClassVar[GameZone] = GameZone.DECK

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): tutored / in_deck,
        with in_deck as sample_count.

        Inputs: summed (nocab_uuid, in_deck, tutored), baseline (None).
        Output: nocab_uuid, LABEL_COLUMN, sample_count.
        Side effects: none. Exceptions: none.

        Example:
            >>> TutorTargetRateMetric.output_from_counts(summed, None)
        """
        return ratio_output(
            summed, cls.KEY_COLUMNS, TUTORED_COLUMN, IN_DECK_COLUMN, cls.LABEL_COLUMN
        )

    def _increments(self, chunk: GameDataChunk) -> npt.NDArray[np.int64]:
        """See CardCountTableMetric._increments(): per deck column, the
        counted games its card was in the deck, and of those, the games
        it was also tutored.

        Inputs: chunk.
        Output: shape (2, deck columns): in_deck, tutored.
        Side effects: none. Exceptions: none.
        """
        deck = chunk.zones[GameZone.DECK]
        in_deck = deck.present() & self._counted_rows(chunk)[:, np.newaxis]

        # Was each deck column's card tutored, under any tutored column?
        tutored = chunk.zones[GameZone.TUTORED].present_for(deck.card_uuids)

        return np.stack(
            [
                in_deck.sum(axis=0, dtype=np.int64),
                (in_deck & tutored).sum(axis=0, dtype=np.int64),
            ]
        )

    def _counted_rows(self, chunk: GameDataChunk) -> npt.NDArray[np.bool_]:
        """Which rows (games) count toward the tallies: every row.

        Inputs: chunk. Output: shape (len(chunk),), all True.
        Side effects: none. Exceptions: none.
        """
        return np.ones(len(chunk), dtype=np.bool_)
