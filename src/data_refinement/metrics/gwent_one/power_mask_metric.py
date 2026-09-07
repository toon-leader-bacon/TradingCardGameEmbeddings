"""Masking metric: a gwent.one card's data-power value.

See src/data_refinement/metrics/masked_field_metric.py for the shared
scan() sequence this fixes MASKED_FIELD/LABEL_VALUES/eligibility for,
and plans/masking_metrics.md for the full design (this class's row in
the gwent_one scope table).

UNIT CARDS ONLY: confirmed against the full page_1.html corpus,
data-power="0" is exactly and only the 323 non-"unit" cards - power
isn't a real value for those types, not a genuinely-zero value worth
predicting.
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


class PowerMaskMetric(MaskedFieldMetric):
    """Card -> data-power, masked, encoded as a classification label
    (str(value), falling back to "OTHER" for anything outside the
    observed range) rather than a regression target."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["power"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        OTHER_LABEL,
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/gwent_one/power_mask.parquet"
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        # Only unit cards carry a real power value - see module
        # docstring.
        return card.raw_content["type"] == _ELIGIBLE_TYPE

    def _label_for_card(self, card: GenericCard) -> str:
        return self._label_or_other(self._raw_field_value(card))
