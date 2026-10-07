"""Per-metric wrappers over PackCardTallyMetric's three take-rate
count tables
(src/data_refinement/metrics/seventeenlands/draft_data/pack_card_tally_metrics.py).

Each is a SeventeenLandsCardLabelDojo (../sliced_dojos.py) that only
names its metric: card in -> take_rate out, from the metric's slice
file.

RankStratifiedTakeRateMetric's and CardTakeRateMetric's extra key
columns (pack_number, pick_number, rank) are not inputs here: a card
appears once per stratum, each row a separate (card -> take_rate)
example. A dojo that also embeds those strata is out of scope.
PickNumberDecayCurveMetric has its own wrapper
(pick_number_decay_curve_dojo.py).
"""

from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metrics import (
    CardTakeRateMetric,
    FirstPickRateMetric,
    RankStratifiedTakeRateMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class CardTakeRateDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted P(picked | in pack, pick_number, pack_number)
    (CardTakeRateMetric)."""

    METRIC = CardTakeRateMetric


class FirstPickRateDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted P(picked | pack 0, pick 0, in pack)
    (FirstPickRateMetric)."""

    METRIC = FirstPickRateMetric


class RankStratifiedTakeRateDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted take rate, one example per (pack, pick, rank)
    stratum (RankStratifiedTakeRateMetric)."""

    METRIC = RankStratifiedTakeRateMetric
