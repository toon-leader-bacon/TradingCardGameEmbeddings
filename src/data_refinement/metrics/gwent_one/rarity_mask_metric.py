"""Masking metric: a gwent.one card's data-rarity value.

See src/data_refinement/metrics/masked_field_metric.py for the shared
scan() sequence this fixes MASKED_FIELD/LABEL_VALUES/eligibility for,
and plans/masking_metrics.md for the full design (this class's row in
the gwent_one scope table).
"""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.metrics.masked_field_metric import MaskedFieldMetric
from src.schema.card import GenericCard
from src.schema.game_id import GameId


class RarityMaskMetric(MaskedFieldMetric):
    """Card -> data-rarity, masked (legendary/epic/rare/common - true
    collector rarity, distinct from data-color). Every gwent.one card
    is eligible - rarity is present on every card."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["rarity"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "legendary",
        "epic",
        "rare",
        "common",
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/gwent_one/rarity_mask.parquet"
    )

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)
