"""Thin wrapper over PoolConditionedPickMetric
(src/data_refinement/metrics/seventeenlands/draft_data/pool_conditioned_pick_metric.py).

A MultiGroupOptionSelectionDojo subclass - adds no behavior of its own,
only configuration. See src/dojos/generic/multi_group_option_selection/
for the generic cell this reuses, and pack_to_pick_choice_set_dojo.py
for the sibling with no pool conditioning.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pool_conditioned_pick_metric import (  # noqa: E501
    PoolConditionedPickMetric,
)
from src.dojos.generic.data_constructors import PoolConditionedPickDataConstructor
from src.dojos.generic.multi_group_option_selection.dojo import (
    MultiGroupOptionSelectionDojo,
)
from src.dojos.generic.option_scoring import OptionScoringHead
from src.dojos.generic.pooling import EmbeddingPooler
from src.schema.holdout import HoldoutSpec


class PoolConditionedPickDojo(MultiGroupOptionSelectionDojo):
    """Pool-so-far + pack options -> which option was picked
    (PoolConditionedPickMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        scoring_head: OptionScoringHead | None = None,
        pooler: EmbeddingPooler | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=path_to_training_data
            or PoolConditionedPickMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=PoolConditionedPickDataConstructor(),
            card_embedding_size=card_embedding_size,
            scoring_head=scoring_head,
            pooler=pooler,
            rng_seed=rng_seed,
        )
