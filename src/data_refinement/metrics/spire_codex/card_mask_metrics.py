"""Single-card masking metrics over the spire_codex (Slay the Spire 2)
CardBinder.

One small class per masked field (see ../scryfall/card_mask_metrics.py
for the shape); the paired dojos (src/dojos/spire_codex/
card_mask_dojos.py) mask the leaking keys. Counts are from the live
binder (577 cards, so these dojos are small).

Curse, Status, Quest, Event and Token cards are skipped where the field
is a category rather than a real value for them: their color and rarity
just repeat that category ("curse"/"Curse"), and they are unplayable.
"""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.metrics.generic.masked_field_metric import (
    OTHER_LABEL,
    MaskedFieldMetric,
)
from src.schema.card import GenericCard
from src.schema.game_id import GameId

_OUTPUT_DIRECTORY = Path("data/metrics/spire_codex")


class CostMaskMetric(MaskedFieldMetric):
    """Card -> energy cost as a class ("0"-"3"; the four 4+ costs are
    OTHER). Classes, not regression: four values cover 99% of eligible
    cards. Eligible: playable (cost >= 0), non-X-cost cards (X cards
    store cost 0 or -1)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.SLAY_THE_SPIRE_2
    MASKED_FIELD: ClassVar[list[str]] = ["cost"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = ("0", "1", "2", "3", OTHER_LABEL)
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "cost_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        is_playable = int(self._raw_field_value(card)) >= 0
        return is_playable and not card.raw_content.get("is_x_cost")

    def _label_for_card(self, card: GenericCard) -> str:
        return self._label_or_other(str(self._raw_field_value(card)))


class CardTypeMaskMetric(MaskedFieldMetric):
    """Card -> Attack, Skill or Power. Eligible: those three types
    (Curse/Status/Quest skipped)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.SLAY_THE_SPIRE_2
    MASKED_FIELD: ClassVar[list[str]] = ["type"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = ("Attack", "Skill", "Power")
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "card_type_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return self._raw_field_value(card) in self.LABEL_VALUES

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)


class RarityMaskMetric(MaskedFieldMetric):
    """Card -> rarity tier. rarity mixes tier and category, so only the
    tiers are eligible: Basic (the starter cards), Common, Uncommon,
    Rare and Ancient."""

    SOURCE_GAME: ClassVar[GameId] = GameId.SLAY_THE_SPIRE_2
    MASKED_FIELD: ClassVar[list[str]] = ["rarity"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "Basic",
        "Common",
        "Uncommon",
        "Rare",
        "Ancient",
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "rarity_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return self._raw_field_value(card) in self.LABEL_VALUES

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)


class ColorMaskMetric(MaskedFieldMetric):
    """Card -> which character's pool it belongs to, or colorless.
    Eligible: those six colors (event/curse/status/token/quest skipped).
    This is the deck-defining field (a run is one character's cards)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.SLAY_THE_SPIRE_2
    MASKED_FIELD: ClassVar[list[str]] = ["color"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "ironclad",
        "silent",
        "defect",
        "necrobinder",
        "regent",
        "colorless",
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "color_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return self._raw_field_value(card) in self.LABEL_VALUES

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)
