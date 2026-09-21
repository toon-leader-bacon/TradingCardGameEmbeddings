"""Generic dojo for the (ragged pack of options in, picked option out) task shape.

A GenericDojo (src/dojos/generic/generic_dojo.py) that supplies this cell's
decoder head and loss; everything else is the shared `Dojo` contract.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.generic.multi_card_option_selection.decoder_head import (
    MultiCardOptionSelectionDecoderHead,
)
from src.dojos.generic.option_scoring import OptionScoringHead
from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec


class MultiCardOptionSelectionDojo(GenericDojo):
    """Dojo for the (ragged pack of options in, picked option out) task shape.

    Callers never build the decoder head or loss themselves; both are this
    cell's own detail. See GenericDojo for the shared constructor inputs,
    side effects and exceptions."""

    def __init__(
        self,
        path_to_training_data: Path,
        data_constructor: DataConstructor,
        card_lookup: CardLookup,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        mod_pipeline: ModPipeline | None = None,
        scoring_head: OptionScoringHead | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.card_embedding_size = card_embedding_size
        super().__init__(
            path_to_training_data=path_to_training_data,
            data_constructor=data_constructor,
            card_lookup=card_lookup,
            holdout=holdout,
            decoder_head=MultiCardOptionSelectionDecoderHead(
                card_embedding_size, scoring_head=scoring_head
            ),
            loss_calculator=PickPredictionCrossEntropyLoss(),
            mod_pipeline=mod_pipeline,
            rng_seed=rng_seed,
        )
