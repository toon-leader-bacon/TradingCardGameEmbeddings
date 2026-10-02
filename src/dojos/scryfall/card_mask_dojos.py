"""Per-metric wrappers over the scryfall (MTG) masking metrics
(src/data_refinement/metrics/scryfall/card_mask_metrics.py).

Each wrapper names its metric and the other raw_content keys that would
give the masked answer away; the paired base masks them all on every
split. Keys added where absent read "[MASK]" on every card, so a key's
presence never hints at the answer.

- cmc: mana_cost spells the mana value out.
- colors: mana_cost, color_identity and color_indicator name the colors.
  oracle_text still mentions colors ("Add {G}"); that is real signal the
  model may use, not a leak to hide.
- card type: power/toughness, loyalty and defense exist only on
  creatures, planeswalkers and battles.
"""

from src.data_refinement.metrics.scryfall.card_mask_metrics import (
    CardTypeMaskMetric,
    CmcRegressionMetric,
    ColorsMaskMetric,
    PowerRegressionMetric,
    RarityMaskMetric,
    ToughnessRegressionMetric,
)
from src.dojos.generic.paired_metric_dojos import (
    MaskedFieldMetricDojo,
    MaskedFieldMultiLabelMetricDojo,
    MaskedFieldRegressionMetricDojo,
)


class CmcRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Card, cmc and mana_cost masked -> predicted mana value."""

    METRIC = CmcRegressionMetric
    EXTRA_MASKED_KEYS = ("mana_cost",)


class CardTypeMaskDojo(MaskedFieldMetricDojo):
    """Card, type_line and type-only stats masked -> predicted card type."""

    METRIC = CardTypeMaskMetric
    EXTRA_MASKED_KEYS = ("power", "toughness", "loyalty", "defense")


class RarityMaskDojo(MaskedFieldMetricDojo):
    """Card, rarity masked -> predicted rarity."""

    METRIC = RarityMaskMetric


class ColorsMaskDojo(MaskedFieldMultiLabelMetricDojo):
    """Card, colors and color-naming keys masked -> P(each of WUBRG)."""

    METRIC = ColorsMaskMetric
    EXTRA_MASKED_KEYS = ("color_identity", "mana_cost", "color_indicator")


class PowerRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Creature, power masked -> predicted power."""

    METRIC = PowerRegressionMetric


class ToughnessRegressionDojo(MaskedFieldRegressionMetricDojo):
    """Creature, toughness masked -> predicted toughness."""

    METRIC = ToughnessRegressionMetric
