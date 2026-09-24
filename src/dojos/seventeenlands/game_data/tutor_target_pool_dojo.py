"""Thin wrapper over TutorTargetPoolMetric
(src/data_refinement/metrics/seventeenlands/game_data/tutor_target_pool_metric.py).

A single SingleCardRegressionDojo subclass - adds no behavior of its
own, only configuration. TutorTargetPoolMetric's id column is
"pool_card_uuid", not "nocab_uuid" (its rows fan out one per pool card,
not one per card in the usual single-card-per-row sense - see that
metric's module docstring), so this wrapper configures
CardAverageDataConstructor's uuid_column accordingly. Its label column
is "tutored" (bool), which CardAverageDataConstructor already casts to
float like any other CardAverageMetric-shaped label.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_pool_metric import (
    TutorTargetPoolMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec


class TutorTargetPoolDojo(SingleCardRegressionDojo):
    """Card -> predicted P(tutored | in pool) (TutorTargetPoolMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or TutorTargetPoolMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                "tutored", uuid_column="pool_card_uuid"
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
            strict_version_check=strict_version_check,
        )
