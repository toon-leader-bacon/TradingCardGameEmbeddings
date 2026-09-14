"""Thin wrapper over CombatAggressionProfileMetric
(src/data_refinement/metrics/seventeenlands/replay_data/combat_aggression_profile_metric.py).

A single MultiCardRegressionDojo subclass - adds no behavior of its
own, only configuration. This metric declares no LABEL_COLUMN ClassVar
(it writes "combat_aggression_profile" as a literal in its own
_output_row() - see that class's source), so this wrapper's
DeckLabelDataConstructor is configured with that same literal rather
than a class attribute reference.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.replay_data.combat_aggression_profile_metric import (  # noqa: E501
    CombatAggressionProfileMetric,
)
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo


class CombatAggressionProfileDojo(MultiCardRegressionDojo):
    """Deck -> predicted average attackers-per-attacking-turn
    (CombatAggressionProfileMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CombatAggressionProfileMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, "combat_aggression_profile"
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
