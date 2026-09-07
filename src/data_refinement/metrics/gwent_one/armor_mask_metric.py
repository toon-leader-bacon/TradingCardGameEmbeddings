"""Masking metric: a gwent.one card's data-armor value.

See src/data_refinement/metrics/masked_field_metric.py for the shared
scan() sequence this fixes MASKED_FIELD/LABEL_VALUES/eligibility for,
and plans/masking_metrics.md for the full design (this class's row in
the gwent_one scope table).

UNIT CARDS ONLY: confirmed against the full page_1.html corpus,
nonzero armor is observed exclusively on data-type="unit" cards.
data-armor="0" stays a valid, common label for eligible unit cards
(most units are legitimately 0 armor) - it is NOT excluded the way
data-power="0"/data-provision="0" are for their own ineligible types.
"""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.metrics.masked_field_metric import (
    OTHER_LABEL,
    MaskedFieldMetric,
)
from src.schema.card import GenericCard
from src.schema.game_id import GameId

_ELIGIBLE_TYPE = "unit"


class ArmorMaskMetric(MaskedFieldMetric):
    """Card -> data-armor, masked, encoded as a classification label
    (str(value), falling back to "OTHER" for anything outside the
    observed values) rather than a regression target."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["armor"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "10",
        OTHER_LABEL,
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/gwent_one/armor_mask.parquet"
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        # Only unit cards ever carry nonzero armor - see module
        # docstring.
        return card.raw_content["type"] == _ELIGIBLE_TYPE

    def _label_for_card(self, card: GenericCard) -> str:
        return self._label_or_other(self._raw_field_value(card))
