"""Single-card masking metrics over the cardvault_fabtcg (Flesh and Blood)
CardBinder.

One small class per masked field (see ../scryfall/card_mask_metrics.py
for the shape); the paired dojos (src/dojos/cardvault_fabtcg/
card_mask_dojos.py) mask the leaking keys. Counts are from the live
binder (5,187 cards).

The leaned card has no classes/talents/types keys: class, talent and
card type are all words in one "typebox" string ("Light Warrior Action -
Attack": talent, class, type, then subtypes after " - "). So the class
and type metrics both mask the whole typebox and read their label off
its words.

Every metric skips the 94 two-faced cards: their "back_face" object
repeats typebox, pitch, color, cost, power and defense for the other
face, so the masked field would still be readable there.
"""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.metrics.cardvault_fabtcg.typebox import (
    typebox_head,
    typebox_words,
)
from src.data_refinement.metrics.generic.masked_field_metric import (
    OTHER_LABEL,
    MaskedFieldMetric,
)
from src.data_refinement.metrics.generic.masked_field_regression_metric import (
    MaskedFieldRegressionMetric,
)
from src.data_refinement.metrics.generic.printed_numbers import is_small_whole_number
from src.schema.card import GenericCard
from src.schema.game_id import GameId

_OUTPUT_DIRECTORY = Path("data/metrics/cardvault_fabtcg")

# Above this a value is not a real stat (one card has cost "99")
_MAX_STAT = 20

# Classes with at least ~60 cards. Bard, Adjudicator, Merchant,
# Shapeshifter and Thief (3-24 cards each) fall to OTHER.
_CLASSES = (
    "Generic",
    "Warrior",
    "Guardian",
    "Mechanologist",
    "Runeblade",
    "Brute",
    "Ninja",
    "Assassin",
    "Illusionist",
    "Wizard",
    "Ranger",
    "Pirate",
    "Necromancer",
)
_RARE_CLASSES = frozenset({"Bard", "Adjudicator", "Merchant", "Shapeshifter", "Thief"})
# A card with talent words only ("Shadow Action") or no class at all
_CLASSLESS_LABEL = "Classless"
# A hybrid card naming two classes ("Assassin / Ninja", "Pirate Necromancer")
_MULTICLASS_LABEL = "Multiclass"

# Two-word card types first, so "Attack Reaction" is not read as Action
_CARD_TYPES_BY_PRIORITY = (
    "Attack Reaction",
    "Defense Reaction",
    "Action",
    "Instant",
    "Equipment",
    "Weapon",
    "Hero",
    "Block",
    "Token",
)


def _is_single_faced(card: GenericCard) -> bool:
    """Whether card has no back_face (whose fields would leak the answer).

    Inputs: card. Output: bool. Side effects: none. Exceptions: none.
    """
    return "back_face" not in card.raw_content


class PitchMaskMetric(MaskedFieldMetric):
    """Card -> pitch 1/2/3 (red/yellow/blue). Eligible: single-faced
    cards with pitch 1-3 (one pitch-4 card is skipped). 927 names come
    in all three pitches with the same text but scaled numbers, which is
    what makes this learnable. A fixed class, not regression: three
    values."""

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD
    MASKED_FIELD: ClassVar[list[str]] = ["pitch"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = ("1", "2", "3")
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "pitch_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return (
            _is_single_faced(card)
            and card.raw_content.get("pitch") in self.LABEL_VALUES
        )

    def _label_for_card(self, card: GenericCard) -> str:
        return self._raw_field_value(card)


class CostRegressionMetric(MaskedFieldRegressionMetric):
    """Card -> resource cost. Eligible: single-faced cards with a
    whole-number cost up to 20 (X costs and the one "99" are skipped).
    Mostly 0-3."""

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD
    MASKED_FIELD: ClassVar[list[str]] = ["cost"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "cost_regression.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_single_faced(card) and is_small_whole_number(
            card.raw_content.get("cost"), _MAX_STAT
        )


class PowerRegressionMetric(MaskedFieldRegressionMetric):
    """Attack/weapon -> power. Eligible: single-faced cards with a
    whole-number power ("*" skipped)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD
    MASKED_FIELD: ClassVar[list[str]] = ["power"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "power_regression.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_single_faced(card) and is_small_whole_number(
            card.raw_content.get("power"), _MAX_STAT
        )


class DefenseRegressionMetric(MaskedFieldRegressionMetric):
    """Card -> defense. Eligible: single-faced cards with a whole-number
    defense ("*" skipped)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD
    MASKED_FIELD: ClassVar[list[str]] = ["defense"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = (
        _OUTPUT_DIRECTORY / "defense_regression.parquet"
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_single_faced(card) and is_small_whole_number(
            card.raw_content.get("defense"), _MAX_STAT
        )


class ClassMaskMetric(MaskedFieldMetric):
    """Card -> its class, read off the typebox words. A card naming no
    class (talent-only cards such as "Shadow Action", events) is
    Classless, one naming two classes is Multiclass, and one naming a
    rare class is OTHER. Eligible: single-faced cards with a typebox."""

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD
    MASKED_FIELD: ClassVar[list[str]] = ["typebox"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        *_CLASSES,
        _CLASSLESS_LABEL,
        _MULTICLASS_LABEL,
        OTHER_LABEL,
    )
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "class_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_single_faced(card) and "typebox" in card.raw_content

    def _label_for_card(self, card: GenericCard) -> str:
        classes = typebox_words(self._raw_field_value(card)) & (
            set(_CLASSES) | _RARE_CLASSES
        )
        if not classes:
            return _CLASSLESS_LABEL
        if len(classes) > 1:
            return _MULTICLASS_LABEL
        return self._label_or_other(next(iter(classes)))


class CardTypeMaskMetric(MaskedFieldMetric):
    """Card -> its card type (Action, Attack Reaction, Equipment, ...),
    read off the typebox by _CARD_TYPES_BY_PRIORITY; Event, Resource,
    Demi-Hero and the other small types are OTHER. Eligible: single-faced
    cards with a typebox."""

    SOURCE_GAME: ClassVar[GameId] = GameId.FLESH_AND_BLOOD
    MASKED_FIELD: ClassVar[list[str]] = ["typebox"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (*_CARD_TYPES_BY_PRIORITY, OTHER_LABEL)
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "card_type_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_single_faced(card) and "typebox" in card.raw_content

    def _label_for_card(self, card: GenericCard) -> str:
        raw_value = self._raw_field_value(card)
        head = typebox_head(raw_value)
        words = typebox_words(raw_value)
        for card_type in _CARD_TYPES_BY_PRIORITY:
            found = card_type in head if " " in card_type else card_type in words
            if found:
                return card_type
        return OTHER_LABEL
