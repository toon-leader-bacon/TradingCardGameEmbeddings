"""Thin per-metric wrappers over PackCardTallyMetric's three take-rate-
shaped concrete subclasses
(src/data_refinement/metrics/seventeenlands/draft_data/pack_card_tally_metrics.py).

Each wrapper is a thin SingleCardRegressionDojo subclass - it adds no
behavior of its own, only configuration: it pulls its paired metric
class's own DEFAULT_OUTPUT_PATH by reference and injects a
CardAverageDataConstructor configured for the shared "take_rate" label
column every PackCardTallyMetric subclass writes. No wrapper
instantiates its paired metric class - that class's own constructor
needs a card_binder/header/source_game a dojo has no reason to
fabricate.

CardAverageDataConstructor, NOT A NEW CONSTRUCTOR: none of these three
metrics subclass CardAverageMetric, but PackCardTallyMetric.finalize()
writes the exact same row shape CardAverageDataConstructor already
consumes (nocab_uuid: str, take_rate: float, sample_count: int) - see
plans/seventeen_lands_dojos.md's "Generic cells reviewed" note on why
that constructor is reused by row shape, not by metric class hierarchy.

RankStratifiedTakeRateMetric's EXTRA KEY COLUMNS (pack_number,
pick_number, rank) ARE IGNORED: CardAverageDataConstructor.build() only
ever reads nocab_uuid and the configured label column, so this wrapper
predicts an unconditioned take rate per card from
RankStratifiedTakeRateMetric's output, the same way it would from
CardTakeRateMetric's - a richer dojo that also embeds those stratifying
columns as input context is future work, not this wrapper's scope (see
plans/seventeen_lands_dojos.md).

SCOPE: PickNumberDecayCurveMetric (a fourth PackCardTallyMetric
subclass) is NOT wrapped here - its output is a per-card vector
(take_rate_by_pick_number), not a scalar, so it doesn't fit this
generic cell. See plans/seventeen_lands_dojos.md's "Not easily
supported" section.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metrics import (
    CardTakeRateMetric,
    FirstPickRateMetric,
    RankStratifiedTakeRateMetric,
)
from src.dojos.generic.data_constructors import CardAverageDataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.single_card_regression.dojo import SingleCardRegressionDojo
from src.schema.holdout import HoldoutSpec


class CardTakeRateDojo(SingleCardRegressionDojo):
    """Card -> predicted P(picked | in pack, pick_number, pack_number)
    (CardTakeRateMetric)."""

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
            or CardTakeRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor("take_rate"),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class FirstPickRateDojo(SingleCardRegressionDojo):
    """Card -> predicted P(pick == card | pack 0, pick 0, card in pack)
    (FirstPickRateMetric)."""

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
            or FirstPickRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor("take_rate"),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )


class RankStratifiedTakeRateDojo(SingleCardRegressionDojo):
    """Card -> predicted take rate, from rank-stratified tallies
    (RankStratifiedTakeRateMetric). See module docstring's note on why
    the rank/pack_number/pick_number stratifying columns are ignored
    here."""

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
            or RankStratifiedTakeRateMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=CardAverageDataConstructor("take_rate"),
            card_embedding_size=card_embedding_size,
            config=DojoConfig(
                rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
