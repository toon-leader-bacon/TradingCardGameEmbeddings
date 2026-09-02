"""Builds TrainingDatum objects for the average-pick-number metric.

Consumes rows shaped like AveragePickNumberMetric's output (nocab_uuid,
average_pick_number - see
src/data_refinement/seventeenlands/draft_game_metrics/average_pick_number_metric.py),
resolves each nocab_uuid against a CardBinder, and pairs the resulting
GenericCard with its average_pick_number label.
"""

from typing import List
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.type_hints import TrainingDatum


class AveragePickNumberDataConstructor:
    """Single card in, single float label out (average_pick_number)."""

    def __init__(self, card_binder: CardBinder):
        self.card_binder = card_binder

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """Convert a chunk of (nocab_uuid, average_pick_number) rows into
        (SingleCardInput, Label) TrainingDatum pairs.

        Rows whose nocab_uuid doesn't resolve to a card, or whose
        average_pick_number doesn't parse as a float, are skipped.
        """
        results: List[TrainingDatum] = []
        for _, row in chunk.iterrows():
            nocab_uuid = row["nocab_uuid"]
            average_pick_number = row["average_pick_number"]

            try:
                card = self.card_binder.get_by_uuid(UUID(nocab_uuid))
            except (TypeError, ValueError):
                continue
            if card is None:
                continue

            try:
                average_pick_number = float(average_pick_number)
            except (TypeError, ValueError):
                continue

            results.append((card, average_pick_number))
        return results
