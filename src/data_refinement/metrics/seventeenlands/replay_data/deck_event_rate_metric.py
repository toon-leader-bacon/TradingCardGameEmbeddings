"""Template Method base for the per-game deck rates: per card in the
user's deck, how often the user did something with it that game (cast
it, discarded it, tutored it) (see replay_data/README.md).

A CardCountTableMetric (../card_count_table_metric.py) over the deck
columns: per deck column, (games in deck, games in deck and the card
appears among the user's FIELDS entries that game). A subclass fixes
OUTPUT_STEM, LABEL_COLUMN and FIELDS.

User-only by design: only the user's half-turns are read (a card in the
user's own deck can only be cast, discarded or tutored by the user, and
there is no oppo_turn_N_cards_tutored column at all).
"""

from typing import ClassVar
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.card_count_table_metric import (
    CardCountTableMetric,
)
from src.data_refinement.metrics.seventeenlands.count_table import ratio_output
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    Actor,
    ReplayDataChunk,
    ReplayField,
)
from src.data_retrieval.seventeenlands.refs import DataType

IN_DECK_COLUMN = "in_deck"
HIT_COLUMN = "hit"


class DeckEventRateMetric(CardCountTableMetric[ReplayDataChunk]):
    """Card -> P(the user's FIELDS name it this game | it is in the deck)."""

    FAMILY: ClassVar[DataType] = DataType.REPLAY
    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = (IN_DECK_COLUMN, HIT_COLUMN)
    FIELDS: ClassVar[tuple[ReplayField, ...]]

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): hit / in_deck, with
        in_deck as sample_count.

        Inputs: summed (nocab_uuid, in_deck, hit), baseline (None).
        Output: nocab_uuid, LABEL_COLUMN, sample_count.
        Side effects: none. Exceptions: none.

        Example:
            >>> CastRateMetric.output_from_counts(summed, None)
        """
        return ratio_output(
            summed, cls.KEY_COLUMNS, HIT_COLUMN, IN_DECK_COLUMN, cls.LABEL_COLUMN
        )

    def _column_card_uuids(self, chunk: ReplayDataChunk) -> tuple[UUID, ...]:
        """See CardCountTableMetric._column_card_uuids(): the deck
        columns.

        Inputs: chunk. Output: tuple of card uuids.
        Side effects: none. Exceptions: none.
        """
        return chunk.deck.card_uuids

    def _increments(self, chunk: ReplayDataChunk) -> npt.NDArray[np.int64]:
        """See CardCountTableMetric._increments(): per deck column, games
        in the deck, and of those, games the user's FIELDS name the card.

        Inputs: chunk.
        Output: shape (2, deck columns): in_deck, hit.
        Side effects: none. Exceptions: none.
        """
        in_deck = chunk.deck.present()
        user_events = chunk.events_for(self.FIELDS).matched().for_actor(Actor.USER)
        named = user_events.rows_naming(chunk.deck_codes, len(chunk))
        return np.stack(
            [
                in_deck.sum(axis=0, dtype=np.int64),
                (in_deck & named).sum(axis=0, dtype=np.int64),
            ]
        )
