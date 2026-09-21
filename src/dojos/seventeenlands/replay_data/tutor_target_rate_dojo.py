"""Thin wrapper over TutorTargetRateMetric (replay-level)
(src/data_refinement/metrics/seventeenlands/replay_data/tutor_target_rate_metric.py).

A single SingleCardRegressionDojo subclass - adds no behavior of its
own, only configuration. This metric declares no LABEL_COLUMN ClassVar
(it writes "tutor_target_rate" as a literal in its own _rate_row() -
see that class's source), so this wrapper's CardAverageDataConstructor
is configured with that same literal rather than a class attribute
reference.

Distinct class from
src/dojos/seventeenlands/game_data/tutor_target_rate_dojo.py's own
TutorTargetRateDojo - each lives in its own source-specific package,
mirroring the two paired metric classes' own naming convention (see
that metric's module docstring).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec


class TutorTargetRateDojo(SingleCardRegressionDojo):
    """Card -> predicted P(tutored at least once in the game | card in
    deck_<name>) (TutorTargetRateMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or TutorTargetRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor("tutor_target_rate"),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
