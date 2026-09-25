"""Thin per-metric wrappers over GameCardAverageMetric's three plain
win-rate-shaped concrete subclasses
(src/data_refinement/metrics/seventeenlands/game_data/game_card_average_metrics.py).

Each wrapper is a thin SingleCardRegressionDojo subclass - it adds no
behavior of its own, only configuration: it pulls its paired metric
class's own LABEL_COLUMN/DEFAULT_OUTPUT_PATH ClassVars by reference
(never duplicated as a literal) and injects a CardAverageDataConstructor
configured for that one label column. No wrapper instantiates its
paired metric class - that class's own constructor needs a
card_binder/header/source_game a dojo has no reason to fabricate.

SCOPE: GameLengthAssociationMetric (this family's fourth concrete
subclass) is wrapped separately in game_length_association_dojo.py,
mirroring that metric's own separate module.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metrics import (
    DrawnWinRateMetric,
    OpeningHandWinRateMetric,
    WinRateWhenInDeckMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec


class WinRateWhenInDeckDojo(SingleCardRegressionDojo):
    """Card -> predicted P(won | card in deck_<name>) (WinRateWhenInDeckMetric)."""

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
            or WinRateWhenInDeckMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                WinRateWhenInDeckMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class OpeningHandWinRateDojo(SingleCardRegressionDojo):
    """Card -> predicted P(won | card in opening_hand_<name>) (OpeningHandWinRateMetric)."""

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
            or OpeningHandWinRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                OpeningHandWinRateMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class DrawnWinRateDojo(SingleCardRegressionDojo):
    """Card -> predicted P(won | card in drawn_<name>) (DrawnWinRateMetric)."""

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
            or DrawnWinRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor(
                DrawnWinRateMetric.LABEL_COLUMN
            ),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
