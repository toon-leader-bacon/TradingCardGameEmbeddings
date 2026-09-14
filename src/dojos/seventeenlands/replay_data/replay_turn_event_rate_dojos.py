"""Thin per-metric wrappers over ReplayTurnEventRateMetric's two
concrete subclasses
(src/data_refinement/metrics/seventeenlands/replay_data/replay_turn_event_rate_metrics.py).

Each wrapper is a thin SingleCardRegressionDojo subclass - it adds no
behavior of its own, only configuration: it pulls its paired metric
class's own LABEL_COLUMN/DEFAULT_OUTPUT_PATH ClassVars by reference and
injects a CardAverageDataConstructor configured for that one label
column. No wrapper instantiates its paired metric class - that class's
own constructor needs a card_binder/header/source_game a dojo has no
reason to fabricate.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.replay_turn_event_rate_metrics import (
    CombatDamagePushThroughRateMetric,
    CombatKillInvolvementRateMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo


class CombatKillInvolvementRateDojo(SingleCardRegressionDojo):
    """Card -> predicted P(a creature died in combat that half-turn |
    card fought that half-turn) (CombatKillInvolvementRateMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CombatKillInvolvementRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CombatKillInvolvementRateMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CombatDamagePushThroughRateDojo(SingleCardRegressionDojo):
    """Card -> predicted P(card in creatures_unblocked | card in
    creatures_attacked) (CombatDamagePushThroughRateMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CombatDamagePushThroughRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CombatDamagePushThroughRateMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
