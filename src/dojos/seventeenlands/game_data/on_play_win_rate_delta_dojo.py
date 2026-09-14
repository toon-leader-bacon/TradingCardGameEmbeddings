"""Thin wrapper over OnPlayWinRateDeltaMetric
(src/data_refinement/metrics/seventeenlands/game_data/on_play_win_rate_delta_metric.py).

A single SingleCardRegressionDojo subclass - adds no behavior of its
own, only configuration. Unlike game_card_average_dojos.py's wrappers,
OnPlayWinRateDeltaMetric declares no LABEL_COLUMN ClassVar (it writes
"on_play_win_rate_delta" as a literal in its own _delta_row() - see
that class's source), so this wrapper's CardAverageDataConstructor is
configured with that same literal rather than a class attribute
reference - there is nothing else to reference by.

NULLABLE LABEL: a card never seen on one side of on_play writes None
for its delta (round-trips as NaN through the parquet float64 column) -
CardAverageDataConstructor._label_as_float() skips a NaN label the same
way it skips an unparseable one (see
src/dojos/generic/data_constructors.py and
plans/seventeen_lands_dojos.md's "Bugfix made while wiring this" note),
so those rows are silently dropped from training rather than corrupting
it with a NaN target.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_delta_metric import (
    OnPlayWinRateDeltaMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo


class OnPlayWinRateDeltaDojo(SingleCardRegressionDojo):
    """Card -> predicted P(won | in deck, on_play) - P(won | in deck,
    on_draw) (OnPlayWinRateDeltaMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or OnPlayWinRateDeltaMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, "on_play_win_rate_delta"
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
