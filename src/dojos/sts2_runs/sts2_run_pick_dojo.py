"""Sts2RunPickDojo - the shared wiring of the four StS2 pick dojos
(card reward, shop purchase, card removal, card upgrade).

A MultiGroupOptionSelectionDojo over [options, deck] with an
OptionPickDataConstructor, split by run. A subclass names its metric and
whether the player may decline; it adds nothing else.

SPLITS: a run yields many rows (about 17 card rewards), so the dojo splits
by run_id (DojoConfig.split_group_column); no run straddles TRAIN and TEST.
"""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts2_runs.pick_choice_metric import PickChoiceMetric
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.multi_group_option_selection.dojo import (
    MultiGroupOptionSelectionDojo,
)
from src.dojos.generic.option_scoring import OptionScoringHead
from src.dojos.generic.pooling import EmbeddingPooler
from src.dojos.sts2_runs.option_pick_data_constructor import (
    OptionPickDataConstructor,
)
from src.schema.holdout import HoldoutSpec


class Sts2RunPickDojo(MultiGroupOptionSelectionDojo):
    """[options, deck] -> the option taken (a PickChoiceMetric's rows).

    Class attributes a subclass sets:
        METRIC: the PickChoiceMetric class whose output is read.
        CAN_SKIP: True when the metric writes NULL picks (card reward).
    """

    METRIC: ClassVar[type[PickChoiceMetric]]
    CAN_SKIP: ClassVar[bool] = False

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
            path_to_training_data: overrides METRIC.DEFAULT_OUTPUT_PATH.
            scoring_head, pooler: as MultiGroupOptionSelectionDojo.
            name: Trainer-facing name and split-file prefix.
            rng_seed: split/shuffle seed; None means unseeded.
            strict_version_check: raise (rather than warn) on a metric
                file built against another CardBinder.
        Output: none (constructor).
        Side effects: as MultiGroupOptionSelectionDojo (may write splits).
        Exceptions: as MultiGroupOptionSelectionDojo.

        Example:
            >>> CardRemovalPickDojo(binder, HoldoutSpec.no_holdout(), 32)
        """
        super().__init__(
            path_to_training_data=path_to_training_data
            or self.METRIC.DEFAULT_OUTPUT_PATH,
            data_constructor=OptionPickDataConstructor(),
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
            can_skip=self.CAN_SKIP,
        )
