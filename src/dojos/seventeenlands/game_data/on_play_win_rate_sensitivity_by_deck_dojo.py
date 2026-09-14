"""Thin wrapper over OnPlayWinRateSensitivityByDeckMetric
(src/data_refinement/metrics/seventeenlands/game_data/on_play_win_rate_sensitivity_by_deck_metric.py).

A single MultiCardRegressionDojo subclass - adds no behavior of its
own, only configuration. This metric declares no LABEL_COLUMN ClassVar
(it writes "on_play_win_rate_sensitivity" as a literal in its own
_sensitivity_row() - see that class's source), so this wrapper's
DeckLabelDataConstructor is configured with that same literal rather
than a class attribute reference.

NULLABLE LABEL: a deck never seen on one side of on_play writes None
for its sensitivity (round-trips as NaN through the parquet float64
column) - DeckLabelDataConstructor.build() now skips a NaN label the
same way it skips an unresolvable deck_uuid (see
src/dojos/generic/data_constructors.py and
plans/seventeen_lands_dojos.md's "Bugfix made while wiring this" note).
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.seventeenlands.game_data.on_play_win_rate_sensitivity_by_deck_metric import (  # noqa: E501
    OnPlayWinRateSensitivityByDeckMetric,
)
from src.dojos.generic.data_constructors import DeckLabelDataConstructor
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo


class OnPlayWinRateSensitivityByDeckDojo(MultiCardRegressionDojo):
    """Deck -> predicted P(won | on_play) - P(won | on_draw)
    (OnPlayWinRateSensitivityByDeckMetric)."""

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
            or OnPlayWinRateSensitivityByDeckMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=DeckLabelDataConstructor(
                card_binder, deck_box, "on_play_win_rate_sensitivity"
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
