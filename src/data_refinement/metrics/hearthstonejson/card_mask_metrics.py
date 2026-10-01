"""Single-card masking metrics over the Hearthstone CardBinder (built from
the newest HearthstoneJSON build; card_binder/hearthstonejson/).

One small class per masked field (see ../scryfall/card_mask_metrics.py
for the shape); the paired dojos (src/dojos/hearthstonejson/
card_mask_dojos.py) mask the leaking keys. Counts are from the live binder
(6,187 cards, build 251951).
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

_OUTPUT_DIRECTORY = Path("data/metrics/hearthstonejson")

# Above this a value is a joke or event card (costs 25, 30 and 100 exist)
_MAX_STAT = 20

_MINION = "MINION"
_SPELL = "SPELL"

_CLASSES = (
    "NEUTRAL",
    "DEATHKNIGHT",
    "DEMONHUNTER",
    "DRUID",
    "HUNTER",
    "MAGE",
    "PALADIN",
    "PRIEST",
    "ROGUE",
    "SHAMAN",
    "WARLOCK",
    "WARRIOR",
)
# A card playable by a listed set of classes ("classes"; 143 cards)
_MULTICLASS_LABEL = "MULTICLASS"

# Minion tribes; "ALL" (Amalgam-style minions) means every one of them
_TRIBES = (
    "BEAST",
    "UNDEAD",
    "ELEMENTAL",
    "MECHANICAL",
    "DRAGON",
    "DEMON",
    "DRAENEI",
    "PIRATE",
    "MURLOC",
    "NAGA",
    "TOTEM",
    "QUILBOAR",
)
_ALL_TRIBES = "ALL"

_SPELL_SCHOOLS = ("SHADOW", "NATURE", "HOLY", "FIRE", "ARCANE", "FROST", "FEL")
# A spell with no school is a real answer, not a missing one
_NO_SCHOOL_LABEL = "NONE"


def _is_minion(card: GenericCard) -> bool:
    """Whether card is a minion. Inputs: card. Output: bool.
    Side effects: none. Exceptions: none."""
    return card.raw_content.get("type") == _MINION


class CostRegressionMetric(MaskedFieldRegressionMetric):
    """Card -> mana cost. Eligible: whole-number costs up to 20 (the
    25/30/100-cost oddities are skipped). Mostly 1-6, so regression."""

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    MASKED_FIELD: ClassVar[list[str]] = ["cost"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "cost_regression.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return is_small_whole_number(card.raw_content.get("cost"), _MAX_STAT)


class AttackRegressionMetric(MaskedFieldRegressionMetric):
    """Minion -> attack. Eligible: minions with a whole-number attack up
    to 20."""

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    MASKED_FIELD: ClassVar[list[str]] = ["attack"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = (
        _OUTPUT_DIRECTORY / "attack_regression.parquet"
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_minion(card) and is_small_whole_number(
            card.raw_content.get("attack"), _MAX_STAT
        )


class HealthRegressionMetric(MaskedFieldRegressionMetric):
    """Minion -> health. Eligible: minions with a whole-number health up
    to 20 (heroes' 30 health and locations' durability-like health are
    not minion health)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    MASKED_FIELD: ClassVar[list[str]] = ["health"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = (
        _OUTPUT_DIRECTORY / "health_regression.parquet"
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_minion(card) and is_small_whole_number(
            card.raw_content.get("health"), _MAX_STAT
        )


class ClassMaskMetric(MaskedFieldMetric):
    """Card -> its class (11 classes or NEUTRAL), or MULTICLASS for a card
    listing several classes. This is the deck-defining field: a deck is
    one class plus neutrals. Eligible: cards with a cardClass or classes
    (40 cards of two recent sets have neither)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    MASKED_FIELD: ClassVar[list[str]] = ["cardClass"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        *_CLASSES,
        _MULTICLASS_LABEL,
        OTHER_LABEL,
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "class_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return "cardClass" in card.raw_content or "classes" in card.raw_content

    def _label_for_card(self, card: GenericCard) -> str:
        if "classes" in card.raw_content:
            return _MULTICLASS_LABEL
        return self._label_or_other(self._raw_field_value(card))


class RarityMaskMetric(MaskedFieldMetric):
    """Card -> rarity (of its original printing). Every card eligible."""

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    MASKED_FIELD: ClassVar[list[str]] = ["rarity"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "FREE",
        "COMMON",
        "RARE",
        "EPIC",
        "LEGENDARY",
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "rarity_mask.parquet"

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)


class CardTypeMaskMetric(MaskedFieldMetric):
    """Card -> MINION / SPELL / WEAPON / LOCATION / HERO. Every card
    eligible."""

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    MASKED_FIELD: ClassVar[list[str]] = ["type"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        _MINION,
        _SPELL,
        "WEAPON",
        "LOCATION",
        "HERO",
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "card_type_mask.parquet"

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)


class RacesMaskMetric(MaskedFieldMultiLabelMetric):
    """Minion -> its tribes (multi-label; about half of minions have
    none, ~150 have two, and "ALL" counts as every tribe)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    MASKED_FIELD: ClassVar[list[str]] = ["races"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = _TRIBES
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "races_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_minion(card)

    def _labels_for_card(self, card: GenericCard) -> frozenset[str]:
        tribes = frozenset(self._raw_field_values(card))
        return frozenset(_TRIBES) if _ALL_TRIBES in tribes else tribes


class SpellSchoolMaskMetric(MaskedFieldMetric):
    """Spell -> its spell school, or NONE (about half of spells have no
    school). Eligible: spells."""

    SOURCE_GAME: ClassVar[GameId] = GameId.HEARTHSTONE
    MASKED_FIELD: ClassVar[list[str]] = ["spellSchool"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        *_SPELL_SCHOOLS,
        _NO_SCHOOL_LABEL,
        OTHER_LABEL,
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = (
        _OUTPUT_DIRECTORY / "spell_school_mask.parquet"
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        return card.raw_content.get("type") == _SPELL

    def _label_for_card(self, card: GenericCard) -> str:
        if "spellSchool" not in card.raw_content:
            return _NO_SCHOOL_LABEL
        return self._label_or_other(self._raw_field_value(card))
