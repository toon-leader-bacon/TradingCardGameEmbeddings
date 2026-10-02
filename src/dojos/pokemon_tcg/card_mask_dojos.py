"""Per-metric wrappers over the pokemon_tcg masking metrics
(src/data_refinement/metrics/pokemon_tcg/card_mask_metrics.py).

Each wrapper names its metric and the other raw_content keys that would
give the masked answer away (see ../scryfall/card_mask_dojos.py):

- types: each attack's energy cost names the Pokemon's own type on 86%
  of Pokemon (12,347 / 14,273), so every attack's "cost" is masked too.
  Pokemon have at most four attacks; a missing attack slot is skipped.
- stage: evolvesFrom exists only on evolved Pokemon and evolvesTo only
  below the top of a line, so both go with subtypes.
"""

from src.data_refinement.metrics.pokemon_tcg.card_mask_metrics import (
    HpRegressionMetric,
    RetreatCostRegressionMetric,
    StageMaskMetric,
    TypesMaskMetric,
    WeaknessMaskMetric,
)
from src.dojos.generic.paired_metric_dojos import (
    MaskedFieldMetricDojo,
    MaskedFieldMultiLabelMetricDojo,
    MaskedFieldRegressionMetricDojo,
)
from src.schema.card_factory import FieldPath

_MAX_ATTACKS = 4
_ATTACK_COST_PATHS: tuple[FieldPath, ...] = tuple(
    ("attacks", index, "cost") for index in range(_MAX_ATTACKS)
)


class HpRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Pokemon, hp masked -> predicted HP."""

    METRIC = HpRegressionMetric


class TypesMaskDojo(MaskedFieldMultiLabelMetricDojo):
    """Pokemon, types and attack costs masked -> P(each energy type)."""

    METRIC = TypesMaskMetric
    EXTRA_MASKED_PATHS = _ATTACK_COST_PATHS


class StageMaskDojo(MaskedFieldMetricDojo):
    """Pokemon, subtypes and evolution links masked -> predicted stage."""

    METRIC = StageMaskMetric
    EXTRA_MASKED_KEYS = ("evolvesFrom", "evolvesTo")


class RetreatCostRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Pokemon, retreat cost masked -> predicted retreat cost."""

    METRIC = RetreatCostRegressionMetric


class WeaknessMaskDojo(MaskedFieldMetricDojo):
    """Pokemon, weaknesses masked -> predicted weakness type."""

    METRIC = WeaknessMaskMetric
