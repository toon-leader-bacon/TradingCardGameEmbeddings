"""Wrapper over TutorChoiceRateMetric
(src/data_refinement/metrics/seventeenlands/game_data/tutor_choice_rate_metric.py).

A SeventeenLandsCardLabelDojo (../sliced_dojos.py): it only names its
metric; the slice file supplies nocab_uuid, LABEL_COLUMN and
sample_count.
"""

from src.data_refinement.metrics.seventeenlands.game_data.tutor_choice_rate_metric import (
    TutorChoiceRateMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsCardLabelDojo


class TutorChoiceRateDojo(SeventeenLandsCardLabelDojo):
    """Card -> predicted P(tutored | in deck, a card was tutored that game)
    (TutorChoiceRateMetric)."""

    METRIC = TutorChoiceRateMetric
