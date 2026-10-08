"""Thin wrapper over CardRewardPickMetric
(src/data_refinement/metrics/sts2_runs/card_reward_pick_metric.py).

A MultiGroupOptionSelectionDojo with can_skip=True: [cards offered, deck so
far] in, the card taken out, or a learned "none of them" option when the
player skipped the reward. Adds no behavior of its own, only configuration.

SKIP LIMIT: the skip option is one learned vector scored against the deck
context, so it cannot see how good the offered cards are; weak skip
accuracy is a limit of that design, not a bug.

SPLITS: a run yields about 17 rows, so the dojo splits by run_id
(DojoConfig.split_group_column); no run straddles TRAIN and TEST.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts2_runs.card_reward_pick_metric import (
    CardRewardPickMetric,
)
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_group_option_selection.dojo import (
    MultiGroupOptionSelectionDojo,
)
from src.dojos.generic.option_scoring import OptionScoringHead
from src.dojos.generic.pooling import EmbeddingPooler
from src.dojos.sts2_runs.card_reward_pick_data_constructor import (
    CardRewardPickDataConstructor,
)
from src.schema.holdout import HoldoutSpec


class CardRewardPickDojo(MultiGroupOptionSelectionDojo):
    """[cards offered, deck so far] -> the card taken, or none
    (CardRewardPickMetric)."""

    METRIC = CardRewardPickMetric

    def __init__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        path_to_training_data: Path | None = None,
        scoring_head: OptionScoringHead | None = None,
        pooler: EmbeddingPooler | None = None,
        name: str | None = None,
        rng_seed: int | None = None,
        strict_version_check: bool = True,
    ) -> None:
        """
        Inputs:
            card_binder: the Slay the Spire 2 binder the metric was built
                from (its version is checked against the metric file).
            holdout: card holdout shared by every dojo in a run.
            card_embedding_size: width of the encoder's card embeddings.
            path_to_training_data: overrides CardRewardPickMetric's
                DEFAULT_OUTPUT_PATH.
            scoring_head, pooler: as MultiGroupOptionSelectionDojo.
            name: Trainer-facing name and split-file prefix.
            rng_seed: split/shuffle seed; None means unseeded.
            strict_version_check: raise (rather than warn) on a metric
                file built against another CardBinder.
        Output: none (constructor).
        Side effects: as MultiGroupOptionSelectionDojo (may write splits).
        Exceptions: as MultiGroupOptionSelectionDojo.

        Example:
            >>> CardRewardPickDojo(binder, HoldoutSpec.no_holdout(), 32)
        """
        super().__init__(
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=CardRewardPickDataConstructor(),
            card_lookup=card_binder,
            holdout=holdout,
            card_embedding_size=card_embedding_size,
            scoring_head=scoring_head,
            pooler=pooler,
            config=DojoConfig(
                name=name,
                rng_seed=rng_seed,
                strict_version_check=strict_version_check,
                split_group_column="run_id",
            ),
            can_skip=True,
        )
