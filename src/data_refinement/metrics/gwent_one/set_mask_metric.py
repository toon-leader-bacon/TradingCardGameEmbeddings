"""Masking metric: a gwent.one card's data-set value.

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


class SetMaskMetric(MaskedFieldMetric):
    """Card -> data-set, masked (baseset + 11 named expansions).
    Every gwent.one card is eligible - set is present on every card."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["set"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "baseset",
        "crimsoncurse",
        "novigrad",
        "cursed toad",
        "iron judgment",
        "price of power",
        "unmillable",
        "uroboros",
        "way of the witcher",
        "master mirror",
        "merchants of ofir",
        "thronebreaker",
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/gwent_one/set_mask.parquet"
    )

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)
