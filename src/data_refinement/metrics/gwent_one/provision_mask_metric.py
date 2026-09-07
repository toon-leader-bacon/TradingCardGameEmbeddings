"""Masking metric: a gwent.one card's data-provision value.

See src/data_refinement/metrics/masked_field_metric.py for the shared
scan() sequence this fixes MASKED_FIELD/LABEL_VALUES/eligibility for,
and plans/masking_metrics.md for the full design (this class's row in
the gwent_one scope table).

STRATAGEM CARDS ARE EXCLUDED: confirmed against the full page_1.html
corpus, data-provision="0" is exactly and only the 12 data-type=
"stratagem" cards - provision isn't a real cost for that type, not a
genuinely-zero value worth predicting.
"""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.metrics.masked_field_metric import (
    OTHER_LABEL,
    MaskedFieldMetric,
)
from src.schema.card import GenericCard
from src.schema.game_id import GameId

_INELIGIBLE_TYPE = "stratagem"


class ProvisionMaskMetric(MaskedFieldMetric):
    """Card -> data-provision, masked, encoded as a classification
    label (str(value), falling back to "OTHER" for anything outside
    the observed dense range) rather than a regression target."""

    SOURCE_GAME: ClassVar[GameId] = GameId.GWENT
    MASKED_FIELD: ClassVar[list[str]] = ["provision"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "0",
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
        "18",
        OTHER_LABEL,
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/gwent_one/provision_mask.parquet"
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        # Stratagem cards don't carry a real provision cost - see
        # module docstring.
        return card.raw_content["type"] != _INELIGIBLE_TYPE

    def _label_for_card(self, card: GenericCard) -> str:
        return self._label_or_other(self._raw_field_value(card))
