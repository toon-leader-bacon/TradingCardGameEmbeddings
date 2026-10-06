"""Thin wrapper over LeaderMaskedFromDeckMetric
(src/data_refinement/metrics/play_gwent/): a DeckCardMaskMetricDojo
subclass (src/dojos/generic/paired_metric_dojos.py) naming only METRIC,
same convention as that module's other thin wrappers - no wrapper
instantiates its paired metric class, since that class's own
constructor needs a card_lookup/deck_box a dojo has no reason to
fabricate.

SCOPE: LeaderMaskedFromDeckMetric is this family's only concrete
metric today.
"""

from src.data_refinement.metrics.play_gwent.leader_masked_from_deck_metric import (
    LeaderMaskedFromDeckMetric,
)
from src.dojos.generic.paired_metric_dojos import DeckCardMaskMetricDojo


class LeaderMaskedFromDeckDojo(DeckCardMaskMetricDojo):
    """Deck (leader masked out) -> leader identity (LeaderMaskedFromDeckMetric).

    label_values=LeaderMaskedFromDeckMetric.LABEL_VALUES unchanged - it
    already includes OTHER_LABEL (see that class's own docstring)."""

    METRIC = LeaderMaskedFromDeckMetric
