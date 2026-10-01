"""Per-metric wrappers over the Hearthstone masking metrics
(src/data_refinement/metrics/hearthstonejson/card_mask_metrics.py).

Each wrapper names its metric and the other raw_content keys that would
give the masked answer away (see ../scryfall/card_mask_dojos.py):

- class: "classes" lists a multiclass card's classes, and runeCost
  exists only on Death Knight cards (156), so both go with cardClass.
- card type: attack/health (minions, weapons, locations, heroes),
  armor (heroes), races (minions only) and spellSchool (spells only) are
  type-specific keys, so their presence alone would name the type.
"""

from src.data_refinement.metrics.hearthstonejson.card_mask_metrics import (
    AttackRegressionMetric,
    CardTypeMaskMetric,
    ClassMaskMetric,
    CostRegressionMetric,
    HealthRegressionMetric,
    RacesMaskMetric,
    RarityMaskMetric,
    SpellSchoolMaskMetric,
)
from src.dojos.generic.paired_metric_dojos import (
    MaskedFieldMetricDojo,
    MaskedFieldMultiLabelMetricDojo,
    MaskedFieldRegressionMetricDojo,
)


class CostRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Card, cost masked -> predicted mana cost."""

    METRIC = CostRegressionMetric


class AttackRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Minion, attack masked -> predicted attack."""

    METRIC = AttackRegressionMetric


class HealthRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Minion, health masked -> predicted health."""

    METRIC = HealthRegressionMetric


class ClassMaskDojo(MaskedFieldMetricDojo):
    """Card, class keys masked -> predicted class."""

    METRIC = ClassMaskMetric
    EXTRA_MASKED_KEYS = ("classes", "runeCost")


class RarityMaskDojo(MaskedFieldMetricDojo):
    """Card, rarity masked -> predicted rarity."""

    METRIC = RarityMaskMetric


class CardTypeMaskDojo(MaskedFieldMetricDojo):
    """Card, type and type-specific keys masked -> predicted card type."""

    METRIC = CardTypeMaskMetric
    EXTRA_MASKED_KEYS = (
        "attack",
        "health",
        "durability",
        "armor",
        "races",
        "spellSchool",
    )


class RacesMaskDojo(MaskedFieldMultiLabelMetricDojo):
    """Minion, races masked -> P(each tribe)."""

    METRIC = RacesMaskMetric


class SpellSchoolMaskDojo(MaskedFieldMetricDojo):
    """Spell, spellSchool masked -> predicted school (or NONE)."""

    METRIC = SpellSchoolMaskMetric
