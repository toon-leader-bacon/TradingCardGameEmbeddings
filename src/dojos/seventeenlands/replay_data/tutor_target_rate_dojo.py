"""Thin wrapper over TutorTargetRateMetric (replay-level)
(src/data_refinement/metrics/seventeenlands/replay_data/tutor_target_rate_metric.py).

A single SingleCardRegressionDojo subclass - adds no behavior of its
own, only configuration. This wrapper's CardAverageDataConstructor is
configured from TutorTargetRateMetric.LABEL_COLUMN, never a duplicated
literal.

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
from src.dojos.generic.dojo_config import DojoConfig
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
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or TutorTargetRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                TutorTargetRateMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
