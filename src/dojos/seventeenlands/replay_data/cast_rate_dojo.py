"""Thin wrapper over CastRateMetric
(src/data_refinement/metrics/seventeenlands/replay_data/cast_rate_metric.py).

A single SingleCardRegressionDojo subclass - adds no behavior of its
own, only configuration. This metric declares no LABEL_COLUMN ClassVar
(it writes "cast_rate" as a literal in its own _rate_row() - see that
class's source), so this wrapper's CardAverageDataConstructor is
configured with that same literal rather than a class attribute
reference.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.cast_rate_metric import (
    CastRateMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec


class CastRateDojo(SingleCardRegressionDojo):
    """Card -> predicted P(cast at least once in the game | card in
    deck_<name>) (CastRateMetric)."""

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
            or CastRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor("cast_rate"),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
