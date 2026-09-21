"""Thin wrapper over GameLengthAssociationMetric
(src/data_refinement/metrics/seventeenlands/game_data/game_length_association_metric.py).

A single SingleCardRegressionDojo subclass - adds no behavior of its
own, only configuration, same convention as
game_card_average_dojos.py's three wrappers (this metric shares that
family's exact output row shape - nocab_uuid/LABEL_COLUMN/sample_count
- despite overriding finalize() itself to subtract a format-wide
baseline; see that class's own docstring).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_length_association_metric import (
    GameLengthAssociationMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec


class GameLengthAssociationDojo(SingleCardRegressionDojo):
    """Card -> predicted (own average num_turns - format-wide average)
    (GameLengthAssociationMetric)."""

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
            or GameLengthAssociationMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                GameLengthAssociationMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
