"""Single-card masking metrics over the scryfall (MTG) CardBinder.

One small class per masked field, each a subclass of a generic masked-
field base (../generic/): which field, which cards are eligible, and the
label. Which OTHER keys leak the answer (and so are masked with it) is
the paired dojo's job: src/dojos/scryfall/card_mask_dojos.py.

Distributions checked on the live binder (34,710 cards; see ./README.md
for the counts behind each choice). Every metric except rarity skips
multi-face cards (card_faces, 898 cards): each face repeats its own
mana_cost, type_line, colors and power, so the masked field would still
be readable inside card_faces.
"""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.metrics.generic.masked_field_metric import (
    OTHER_LABEL,
    MaskedFieldMetric,
)
from src.data_refinement.metrics.generic.masked_field_multi_label_metric import (
    MaskedFieldMultiLabelMetric,
)
from src.data_refinement.metrics.generic.masked_field_regression_metric import (
    MaskedFieldRegressionMetric,
)
from src.data_refinement.metrics.generic.printed_numbers import is_small_whole_number
from src.schema.card import GenericCard
from src.schema.game_id import GameId

_OUTPUT_DIRECTORY = Path("data/metrics/scryfall")

# Above this a value is a joke card (Gleemax cmc 1,000,000; a 99/99)
_MAX_STAT = 20

# Card types in priority order: a card's label is the first one its
# type_line names ("Artifact Creature" -> Creature, "Artifact Land" ->
# Land). Battle (2 cards) and the rest fall to OTHER.
_CARD_TYPES_BY_PRIORITY = (
    "Creature",
    "Planeswalker",
    "Land",
    "Instant",
    "Sorcery",
    "Artifact",
    "Enchantment",
)


def _is_single_faced(card: GenericCard) -> bool:
    """Whether card has no card_faces (whose faces would leak the field).

    Inputs: card. Output: bool. Side effects: none. Exceptions: none.
    """
    return "card_faces" not in card.raw_content


class CmcRegressionMetric(MaskedFieldRegressionMetric):
    """Card -> mana value (cmc). Eligible: single-faced cards with a
    mana_cost (lands and cost-less suspend cards always have cmc 0 and
    no mana_cost, so they would be free answers) and a whole-number cmc
    up to 20 (Gleemax and the one cmc 0.5 Un-card are skipped). 19
    distinct values over 0..16, so regression keeps the ordering."""

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    MASKED_FIELD: ClassVar[list[str]] = ["cmc"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "cmc_regression.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return (
            _is_single_faced(card)
            and "mana_cost" in card.raw_content
            and is_small_whole_number(self._raw_field_value(card), _MAX_STAT)
        )


class CardTypeMaskMetric(MaskedFieldMetric):
    """Card -> its main card type, read off type_line by
    _CARD_TYPES_BY_PRIORITY. Eligible: single-faced cards."""

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    MASKED_FIELD: ClassVar[list[str]] = ["type_line"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (*_CARD_TYPES_BY_PRIORITY, OTHER_LABEL)
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "card_type_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_single_faced(card)

    def _label_for_card(self, card: GenericCard) -> str:
        # Supertypes and card types come before the " — " subtypes
        card_types = self._raw_field_value(card).split(" — ")[0].split()
        for card_type in _CARD_TYPES_BY_PRIORITY:
            if card_type in card_types:
                return card_type
        return OTHER_LABEL


class RarityMaskMetric(MaskedFieldMetric):
    """Card -> rarity. Every card is eligible (no other field names the
    rarity). special (63) and bonus (9) fall to OTHER."""

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    MASKED_FIELD: ClassVar[list[str]] = ["rarity"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "common",
        "uncommon",
        "rare",
        "mythic",
        OTHER_LABEL,
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "rarity_mask.parquet"

    def _label_for_card(self, card: GenericCard) -> str:
        return self._label_or_other(self._raw_field_value(card))


class ColorsMaskMetric(MaskedFieldMultiLabelMetric):
    """Card -> its set of colors (multi-label over WUBRG; colorless is
    the empty set, about 14% of cards). Eligible: single-faced cards."""

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    MASKED_FIELD: ClassVar[list[str]] = ["colors"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = ("W", "U", "B", "R", "G")
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "colors_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_single_faced(card)

    def _labels_for_card(self, card: GenericCard) -> frozenset[str]:
        return frozenset(self._raw_field_values(card))


class PowerRegressionMetric(MaskedFieldRegressionMetric):
    """Creature -> power. Eligible: single-faced cards whose power is a
    whole number 0..20 ("*", "1+*", half-points and the 99 are not)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    MASKED_FIELD: ClassVar[list[str]] = ["power"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "power_regression.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_single_faced(card) and is_small_whole_number(
            card.raw_content.get("power"), _MAX_STAT
        )


class ToughnessRegressionMetric(MaskedFieldRegressionMetric):
    """Creature -> toughness. Eligibility as PowerRegressionMetric."""

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG
    MASKED_FIELD: ClassVar[list[str]] = ["toughness"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = (
        _OUTPUT_DIRECTORY / "toughness_regression.parquet"
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_single_faced(card) and is_small_whole_number(
            card.raw_content.get("toughness"), _MAX_STAT
        )
