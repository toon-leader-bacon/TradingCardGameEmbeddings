"""Masking metric: a gwent.one card's data-color value.

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


class ColorMaskMetric(MaskedFieldMetric):
    """Card -> data-color, masked (gold/bronze/leader - power-budget
    tier, distinct from data-rarity). Every gwent.one card is
    eligible - color is present on every card."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["color"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = ("gold", "bronze", "leader")
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/gwent_one/color_mask.parquet"
    )

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)
