"""Thin wrapper over PackToPickChoiceSetMetric
(src/data_refinement/metrics/seventeenlands/draft_data/pack_to_pick_choice_set_metric.py).

A MultiCardOptionSelectionDojo subclass - adds no behavior of its own,
only configuration. See src/dojos/generic/multi_card_option_selection/
for the generic cell this reuses, and pool_conditioned_pick_dojo.py for
the sibling that additionally conditions on the drafter's pool.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pack_to_pick_choice_set_metric import (  # noqa: E501
    PackToPickChoiceSetMetric,
)
from src.dojos.generic.data_constructors import PackToPickChoiceSetDataConstructor
from src.dojos.generic.multi_card_option_selection.dojo import (
    MultiCardOptionSelectionDojo,
)
from src.dojos.generic.option_scoring import OptionScoringHead


class PackToPickChoiceSetDojo(MultiCardOptionSelectionDojo):
    """Pack options -> which one was picked (PackToPickChoiceSetMetric)."""

    def __init__(
        self,
        card_binder: CardBinder,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        scoring_head: OptionScoringHead | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(
            path_to_training_data=path_to_training_data
            or PackToPickChoiceSetMetric.DEFAULT_OUTPUT_PATH,
            data_constructor=PackToPickChoiceSetDataConstructor(card_binder),
            card_embedding_size=card_embedding_size,
            scoring_head=scoring_head,
            rng_seed=rng_seed,
        )
