"""Masking metric: a gwent.one card's data-faction value.

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


class FactionMaskMetric(MaskedFieldMetric):
    """Card -> data-faction, masked. Every gwent.one card is eligible -
    faction is present on every card, confirmed against the full
    page_1.html corpus (1,260/1,260 rows)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["faction"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "neutral",
        "syndicate",
        "monster",
        "scoiatael",
        "nilfgaard",
        "northern_realms",
        "skellige",
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/gwent_one/faction_mask.parquet"
    )

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)
