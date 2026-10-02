"""Per-metric wrappers over the cardvault_fabtcg (Flesh and Blood)
masking metrics (src/data_refinement/metrics/cardvault_fabtcg/
card_mask_metrics.py).

Each wrapper names its metric and the other raw_content keys that would
give the masked answer away (see ../scryfall/card_mask_dojos.py):

- pitch: color is pitch in other words (red 1, yellow 2, blue 3 on
  every card that has both), so it is masked too.
- class and card type: both read off the typebox, which is masked
  whole.

Two-faced cards (whose back_face repeats these fields) are not in the
metrics at all, so nothing here masks back_face.
"""

from src.data_refinement.metrics.cardvault_fabtcg.card_mask_metrics import (
    CardTypeMaskMetric,
    ClassMaskMetric,
    CostRegressionMetric,
    DefenseRegressionMetric,
    PitchMaskMetric,
    PowerRegressionMetric,
)
from src.dojos.generic.paired_metric_dojos import (
    MaskedFieldMetricDojo,
    MaskedFieldRegressionMetricDojo,
)


class PitchMaskDojo(MaskedFieldMetricDojo):
    """Card, pitch and color masked -> predicted pitch."""

    METRIC = PitchMaskMetric
    EXTRA_MASKED_KEYS = ("color",)


class CostRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Card, cost masked -> predicted cost."""

    METRIC = CostRegressionMetric


class PowerRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Card, power masked -> predicted power."""

    METRIC = PowerRegressionMetric


class DefenseRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Card, defense masked -> predicted defense."""

    METRIC = DefenseRegressionMetric


class ClassMaskDojo(MaskedFieldMetricDojo):
    """Card, typebox masked -> predicted class."""

    METRIC = ClassMaskMetric


class CardTypeMaskDojo(MaskedFieldMetricDojo):
    """Card, typebox masked -> predicted card type."""

    METRIC = CardTypeMaskMetric
