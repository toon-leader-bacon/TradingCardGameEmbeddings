"""Single-card masking metrics over the pokemon_tcg CardBinder.

One small class per masked field (see ../scryfall/card_mask_metrics.py
for the shape). Every metric here covers Pokemon cards only (supertype
"Pokémon", 14,273 of 16,803 cards): Trainer and Energy cards have no
HP, type, stage, retreat cost or weakness. The paired dojos
(src/dojos/pokemon_tcg/card_mask_dojos.py) mask the leaking keys.

Not built, on purpose: supertype (hp/attacks vs rules give it away on
every card) and trainer subtype (about 85% of Trainers carry reminder
text in "rules" that names it, e.g. "You may play only 1 Supporter
card", and rules is also the card's whole effect text).
"""

from pathlib import Path
from typing import ClassVar, cast

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
from src.schema.card import GenericCard
from src.schema.game_id import GameId

_OUTPUT_DIRECTORY = Path("data/metrics/pokemon_tcg")
_POKEMON_SUPERTYPE = "Pokémon"

# The eleven energy types, used for both a card's types and its weakness
_ENERGY_TYPES = (
    "Grass",
    "Fire",
    "Water",
    "Lightning",
    "Psychic",
    "Fighting",
    "Darkness",
    "Metal",
    "Fairy",
    "Dragon",
    "Colorless",
)

# A Pokemon's evolution stage, as named in its subtypes
_STAGES = ("Basic", "Stage 1", "Stage 2")


def _is_pokemon(card: GenericCard) -> bool:
    """Whether card is a Pokemon card (not a Trainer or Energy).

    Inputs: card. Output: bool. Side effects: none. Exceptions: none.
    """
    return card.raw_content.get("supertype") == _POKEMON_SUPERTYPE


class HpRegressionMetric(MaskedFieldRegressionMetric):
    """Pokemon -> HP. 10..380 in steps of 10 (36 values), so regression.
    Every Pokemon card has a numeric hp."""

    SOURCE_GAME: ClassVar[GameId] = GameId.POKEMON
    MASKED_FIELD: ClassVar[list[str]] = ["hp"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "hp_regression.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_pokemon(card) and str(card.raw_content.get("hp", "")).isdigit()


class TypesMaskMetric(MaskedFieldMultiLabelMetric):
    """Pokemon -> its energy types (multi-label; 113 cards are dual-type)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.POKEMON
    MASKED_FIELD: ClassVar[list[str]] = ["types"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = _ENERGY_TYPES
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "types_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_pokemon(card)

    def _labels_for_card(self, card: GenericCard) -> frozenset[str]:
        return frozenset(self._raw_field_values(card))


class StageMaskMetric(MaskedFieldMetric):
    """Pokemon -> evolution stage: Basic, Stage 1 or Stage 2 from its
    subtypes; stages outside that ladder (VMAX, VSTAR, BREAK, Level-Up,
    MEGA alone, ...; about 3%) are OTHER."""

    SOURCE_GAME: ClassVar[GameId] = GameId.POKEMON
    MASKED_FIELD: ClassVar[list[str]] = ["subtypes"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (*_STAGES, OTHER_LABEL)
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "stage_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_pokemon(card) and "subtypes" in card.raw_content

    def _label_for_card(self, card: GenericCard) -> str:
        # subtypes is a list, not the str _raw_field_value() is typed as
        subtypes = cast(list[str], self._raw_field_value(card))
        for stage in _STAGES:
            if stage in subtypes:
                return stage
        return OTHER_LABEL


class RetreatCostRegressionMetric(MaskedFieldRegressionMetric):
    """Pokemon -> converted retreat cost (0..5, mostly 1-2), as
    regression since it is an ordered count. The 780 Pokemon without
    the key are skipped: the source omits it rather than writing 0
    (only one card has an explicit 0), so their true value is unknown."""

    SOURCE_GAME: ClassVar[GameId] = GameId.POKEMON
    MASKED_FIELD: ClassVar[list[str]] = ["convertedRetreatCost"]
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = (
        _OUTPUT_DIRECTORY / "retreat_cost_regression.parquet"
    )

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_pokemon(card) and "convertedRetreatCost" in card.raw_content


class WeaknessMaskMetric(MaskedFieldMetric):
    """Pokemon -> the type of its first weakness. Eligible: Pokemon with
    at least one weakness (378 have none; 30 have two, labeled by the
    first)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.POKEMON
    MASKED_FIELD: ClassVar[list[str]] = ["weaknesses"]
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (*_ENERGY_TYPES, OTHER_LABEL)
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _OUTPUT_DIRECTORY / "weakness_mask.parquet"

    def _is_eligible(self, card: GenericCard) -> bool:
        return _is_pokemon(card) and bool(card.raw_content.get("weaknesses"))

    def _label_for_card(self, card: GenericCard) -> str:
        # weaknesses is a list of {type, value}, not the str
        # _raw_field_value() is typed as
        weaknesses = cast(list[dict[str, str]], self._raw_field_value(card))
        return self._label_or_other(weaknesses[0]["type"])
