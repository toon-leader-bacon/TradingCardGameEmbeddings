"""Thin wrapper over AverageTurnCastMetric
(src/data_refinement/metrics/seventeenlands/replay_data/average_turn_cast_metric.py).

A single SingleCardRegressionDojo subclass - adds no behavior of its
own, only configuration. This metric declares no LABEL_COLUMN ClassVar
(it writes "average_turn_cast" as a literal in its own _cast_row() -
see that class's source), so this wrapper's CardAverageDataConstructor
is configured with that same literal rather than a class attribute
reference.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.average_turn_cast_metric import (
    AverageTurnCastMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo


class AverageTurnCastDojo(SingleCardRegressionDojo):
    """Card -> predicted average turn number cast on (AverageTurnCastMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or AverageTurnCastMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, "average_turn_cast"
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
