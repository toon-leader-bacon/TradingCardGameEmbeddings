"""Wrapper over CombatAggressionProfileMetric
(src/data_refinement/metrics/seventeenlands/replay_data/combat_aggression_profile_metric.py).

A SeventeenLandsDeckRegressionDojo (../sliced_dojos.py): deck in ->
combat_aggression_profile out, each row's deck_uuid looked up in the
replay_data family deck box.
"""

from src.data_refinement.metrics.seventeenlands.replay_data.combat_aggression_profile_metric import (  # noqa: E501
    CombatAggressionProfileMetric,
)
from src.dojos.seventeenlands.sliced_dojos import SeventeenLandsDeckRegressionDojo


class CombatAggressionProfileDojo(SeventeenLandsDeckRegressionDojo):
    """Deck -> predicted average attackers per attacking user half-turn
    (CombatAggressionProfileMetric)."""

    METRIC = CombatAggressionProfileMetric
