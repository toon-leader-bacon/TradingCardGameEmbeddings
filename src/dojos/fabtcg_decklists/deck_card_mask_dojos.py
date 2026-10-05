"""Thin wrapper over HeroMaskedFromDeckMetric
(src/data_refinement/metrics/fabtcg_decklists/), the FaB twin of
../play_gwent/deck_card_mask_dojos.py's LeaderMaskedFromDeckDojo - both
are DeckCardMaskMetricDojo subclasses
(src/dojos/generic/paired_metric_dojos.py) naming only METRIC, same
convention as that module's other thin wrappers. Its rows point into
the published FaB deck box, which the catalog passes in (read-only).
"""

from src.data_refinement.metrics.fabtcg_decklists.hero_masked_from_deck_metric import (
    HeroMaskedFromDeckMetric,
)
from src.dojos.generic.paired_metric_dojos import DeckCardMaskMetricDojo


class HeroMaskedFromDeckDojo(DeckCardMaskMetricDojo):
    """FaB deck (hero masked out) -> hero identity, over HERO_NAMES plus
    OTHER (HeroMaskedFromDeckMetric.LABEL_VALUES)."""

    METRIC = HeroMaskedFromDeckMetric
