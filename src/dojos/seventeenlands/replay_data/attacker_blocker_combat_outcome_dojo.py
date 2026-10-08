"""Thin wrapper over AttackerBlockerCombatOutcomeMetric
(src/data_refinement/metrics/seventeenlands/replay_data/attacker_blocker_combat_outcome_metric.py).

A MultiGroupRegressionDojo subclass - adds no behavior of its own, only
configuration. See src/dojos/generic/multi_group_regression/ for the
generic cell this reuses.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.data_slice import (
    SeventeenLandsSlice,
)
from src.data_refinement.metrics.seventeenlands.replay_data.attacker_blocker_combat_outcome_metric import (  # noqa: E501
    AttackerBlockerCombatOutcomeMetric,
)
from src.dojos.loss.regression_objective import RegressionObjective
from src.dojos.generic.data_constructors import (
    AttackerBlockerCombatOutcomeDataConstructor,
)
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_group_regression.dojo import MultiGroupRegressionDojo
from src.dojos.generic.pooling import EmbeddingPooler
from src.dojos.seventeenlands.sliced_dojos import (
    ALL_DATA,
    seventeenlands_training_path,
)
from src.schema.holdout import HoldoutSpec


class AttackerBlockerCombatOutcomeDojo(MultiGroupRegressionDojo):
    """Attacker group + blocker group -> net kill-count delta
    (AttackerBlockerCombatOutcomeMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        data_slice: SeventeenLandsSlice = ALL_DATA,
        path_to_training_data: Path | None = None,
        pooler: EmbeddingPooler | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
        objective: RegressionObjective | None = None,
    ) -> None:
        super().__init__(
            card_lookup=card_binder,
            holdout=holdout,
            path_to_training_data=seventeenlands_training_path(
                AttackerBlockerCombatOutcomeMetric, data_slice, path_to_training_data
            ),
            data_constructor=AttackerBlockerCombatOutcomeDataConstructor(),
            card_embedding_size=card_embedding_size,
            pooler=pooler,
            objective=objective,
            config=DojoConfig(
                name=name, rng_seed=rng_seed, strict_version_check=strict_version_check
            ),
        )
