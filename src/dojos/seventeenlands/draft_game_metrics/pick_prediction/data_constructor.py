"""Builds TrainingDatum objects for the pick-prediction metric.

Consumes rows shaped like PickedVHeldVPackMetric's output (picked_card_uuid,
pack_cards_uuids, pool_cards_uuids - see
src/data_refinement/seventeenlands/draft_game_metrics/picked_v_held_v_pack_metric.py),
resolves every uuid against a CardBinder, and pairs a MultiGroupInput
(pack cards, pool cards) with the index of the picked card within the pack.

Assumed CSV cell encoding: pack_cards_uuids/pool_cards_uuids are each a
JSON-encoded list of nocab_uuid strings, same convention used elsewhere in
this refactor for list-shaped CSV cells (see the old
ExampleDataConstructor.build_multi_group_example this replaces).
"""

import json
from typing import List
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.type_hints import TrainingDatum


class PickPredictionDataConstructor:
    """Multi-group in ([pack_cards, pool_cards]), single int label out (the
    index of the picked card within pack_cards)."""

    def __init__(self, card_binder: CardBinder):
        self.card_binder = card_binder

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """Convert a chunk of (picked_card_uuid, pack_cards_uuids,
        pool_cards_uuids) rows into (MultiGroupInput, Label) TrainingDatum
        pairs.

        Rows that fail to resolve every card, or whose picked card isn't
        actually present in that row's pack, are skipped.
        """
        results: List[TrainingDatum] = []
        for _, row in chunk.iterrows():
            picked_card_uuid = row["picked_card_uuid"]

            try:
                pack_uuid_list = json.loads(row["pack_cards_uuids"])
                pool_uuid_list = json.loads(row["pool_cards_uuids"])
            except (TypeError, ValueError):
                continue

            if picked_card_uuid not in pack_uuid_list:
                # The true label must be one of the pack's own candidates.
                continue
            picked_index = pack_uuid_list.index(picked_card_uuid)

            pack_cards = [self.card_binder.get_by_uuid(UUID(uuid)) for uuid in pack_uuid_list]
            pool_cards = [self.card_binder.get_by_uuid(UUID(uuid)) for uuid in pool_uuid_list]
            if not all(pack_cards) or not all(pool_cards):
                continue

            results.append(([pack_cards, pool_cards], picked_index))
        return results
