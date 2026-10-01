"""Per-metric wrappers over the spire_codex (Slay the Spire 2) masking
metrics (src/data_refinement/metrics/spire_codex/card_mask_metrics.py).

Each wrapper names its metric and the other raw_content keys that would
give the masked answer away (see ../scryfall/card_mask_dojos.py):

- cost: "upgrade" holds the upgraded cost on 57 cards (usually the base
  cost minus one), so upgrade.cost is masked with cost where present;
  the rest of upgrade (added keywords, removed exhaust) stays visible.
"""

from src.data_refinement.metrics.spire_codex.card_mask_metrics import (
    CardTypeMaskMetric,
    ColorMaskMetric,
    CostMaskMetric,
    RarityMaskMetric,
)
from src.dojos.generic.paired_metric_dojos import MaskedFieldMetricDojo


class CostMaskDojo(MaskedFieldMetricDojo):
    """Card, cost and upgraded cost masked -> predicted cost class."""

    METRIC = CostMaskMetric
    EXTRA_MASKED_PATHS = (("upgrade", "cost"),)


class CardTypeMaskDojo(MaskedFieldMetricDojo):
    """Card, type masked -> predicted Attack/Skill/Power."""

    METRIC = CardTypeMaskMetric


class RarityMaskDojo(MaskedFieldMetricDojo):
    """Card, rarity masked -> predicted rarity tier."""

    METRIC = RarityMaskMetric


class ColorMaskDojo(MaskedFieldMetricDojo):
    """Card, color masked -> predicted character (or colorless)."""

    METRIC = ColorMaskMetric
