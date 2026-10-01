"""Thin wrappers over HeldOutDeckCardMetricDojo, one per
src/data_refinement/metrics/final_decks/held_out_card_metrics.py metric.
Each reads its game's published deck box (data/final/decks/<game>.db).
"""

from src.data_refinement.metrics.final_decks.held_out_card_metrics import (
    DominionHeldOutCardMetric,
    FleshAndBloodHeldOutCardMetric,
    GwentHeldOutCardMetric,
    MtgHeldOutCardMetric,
    PokemonHeldOutCardMetric,
    SlayTheSpire2HeldOutCardMetric,
)
from src.dojos.generic.paired_metric_dojos import HeldOutDeckCardMetricDojo


class PokemonHeldOutCardDojo(HeldOutDeckCardMetricDojo):
    """Pokemon deck minus one card -> the held-out card among candidates."""

    METRIC = PokemonHeldOutCardMetric


class FleshAndBloodHeldOutCardDojo(HeldOutDeckCardMetricDojo):
    """FaB deck minus one card -> the held-out card among candidates."""

    METRIC = FleshAndBloodHeldOutCardMetric


class GwentHeldOutCardDojo(HeldOutDeckCardMetricDojo):
    """Gwent deck minus one card -> the held-out card among candidates."""

    METRIC = GwentHeldOutCardMetric


class DominionHeldOutCardDojo(HeldOutDeckCardMetricDojo):
    """Dominion deck minus one card -> the held-out card among candidates."""

    METRIC = DominionHeldOutCardMetric


class SlayTheSpire2HeldOutCardDojo(HeldOutDeckCardMetricDojo):
    """StS2 deck minus one card -> the held-out card among candidates."""

    METRIC = SlayTheSpire2HeldOutCardMetric


class MtgHeldOutCardDojo(HeldOutDeckCardMetricDojo):
    """MTG deck minus one card -> the held-out card among candidates."""

    METRIC = MtgHeldOutCardMetric
