"""GameCardCountTableMetric - the game_data per-card count tables'
shared base: a CardCountTableMetric (../card_count_table_metric.py) over
one game_data zone's columns (ZONE).

Subclasses: GameCardAverageMetric (and its four concretes),
OnPlayWinRateDeltaMetric, TutorTargetRateMetric (and
TutorChoiceRateMetric).
"""

from typing import ClassVar
from uuid import UUID

from src.data_refinement.metrics.seventeenlands.card_count_table_metric import (
    CardCountTableMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_retrieval.seventeenlands.refs import DataType


class GameCardCountTableMetric(CardCountTableMetric[GameDataChunk]):
    """Per card in ZONE: COUNT_COLUMNS tallies, summed over a CSV."""

    FAMILY: ClassVar[DataType] = DataType.GAME
    ZONE: ClassVar[GameZone]

    def _column_card_uuids(self, chunk: GameDataChunk) -> tuple[UUID, ...]:
        """See CardCountTableMetric._column_card_uuids(): ZONE's columns.

        Inputs: chunk. Output: tuple of card uuids.
        Side effects: none. Exceptions: none.
        """
        return chunk.zones[self.ZONE].card_uuids
