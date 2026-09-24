"""Thin wrapper over TurnsToGameEndAfterCastMetric
(src/data_refinement/metrics/seventeenlands/replay_data/turns_to_game_end_after_cast_metric.py).

A single SingleCardRegressionDojo subclass - adds no behavior of its
own, only configuration. This metric declares no LABEL_COLUMN ClassVar
(it writes "turns_to_game_end_after_cast" as a literal in its own
_delta_row() - see that class's source), so this wrapper's
CardAverageDataConstructor is configured with that same literal rather
than a class attribute reference.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.turns_to_game_end_after_cast_metric import (  # noqa: E501
    TurnsToGameEndAfterCastMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec


class TurnsToGameEndAfterCastDojo(SingleCardRegressionDojo):
    """Card -> predicted average (num_turns - first cast turn)
    (TurnsToGameEndAfterCastMetric)."""

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
            or TurnsToGameEndAfterCastMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor("turns_to_game_end_after_cast"),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
            strict_version_check=strict_version_check,
        )
