"""Thin per-metric wrappers over CardAverageMetric's nine concrete
subclasses (src/data_refinement/metrics/sts_gg/card_average_metrics.py).

Each wrapper is a thin SingleCardRegressionDojo subclass - it adds no
behavior of its own, only configuration: it pulls its paired metric
class's own LABEL_COLUMN/DEFAULT_OUTPUT_PATH ClassVars by reference
(never duplicated as a literal) and passes them to
SingleCardRegressionDojo.__init__ via a CardAverageDataConstructor
configured for that one label column. No wrapper instantiates its
paired metric class - that class's own constructor needs a card_binder
a dojo has no reason to fabricate.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts_gg.card_average_metrics import (
    CardDeckSizeMetric,
    CardElitesKilledMetric,
    CardFloorsClearedMetric,
    CardRelicCountMetric,
    CardTotalCardsPickedMetric,
    CardTotalCombatsMetric,
    CardTotalDamageTakenMetric,
    CardTotalTurnsMetric,
    CardWinRateMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo


class CardRelicCountDojo(SingleCardRegressionDojo):
    """Card -> predicted average relicCount (CardRelicCountMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardRelicCountMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CardRelicCountMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CardTotalDamageTakenDojo(SingleCardRegressionDojo):
    """Card -> predicted average stats.totalDamageTaken (CardTotalDamageTakenMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardTotalDamageTakenMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CardTotalDamageTakenMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CardDeckSizeDojo(SingleCardRegressionDojo):
    """Card -> predicted average deckSize (CardDeckSizeMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardDeckSizeMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CardDeckSizeMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CardTotalCardsPickedDojo(SingleCardRegressionDojo):
    """Card -> predicted average stats.totalCardsPicked (CardTotalCardsPickedMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardTotalCardsPickedMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CardTotalCardsPickedMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CardTotalTurnsDojo(SingleCardRegressionDojo):
    """Card -> predicted average stats.totalTurns (CardTotalTurnsMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardTotalTurnsMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CardTotalTurnsMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CardElitesKilledDojo(SingleCardRegressionDojo):
    """Card -> predicted average stats.elitesKilled (CardElitesKilledMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardElitesKilledMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CardElitesKilledMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CardFloorsClearedDojo(SingleCardRegressionDojo):
    """Card -> predicted average stats.floorsCleared (CardFloorsClearedMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardFloorsClearedMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CardFloorsClearedMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CardTotalCombatsDojo(SingleCardRegressionDojo):
    """Card -> predicted average stats.totalCombats (CardTotalCombatsMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardTotalCombatsMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CardTotalCombatsMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )


class CardWinRateDojo(SingleCardRegressionDojo):
    """Card -> predicted P(win | card in final deck) (CardWinRateMetric).

    A regression cell despite the label being a rate in [0, 1] - see
    CardAverageMetric's WIN RATE IS AN AVERAGE docstring note for why a
    win/loss indicator is folded through the same averaging machinery
    as every other CardAverageMetric subclass, rather than treated as a
    two-class classification problem.
    """

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or CardWinRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                card_binder, CardWinRateMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            rng_seed=rng_seed,
        )
