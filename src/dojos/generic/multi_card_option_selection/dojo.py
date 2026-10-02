"""Generic dojo for the (ragged pack of options in, picked option out) task shape.

A GenericDojo (src/dojos/generic/generic_dojo.py) that supplies this cell's
decoder head and loss; everything else is the shared `Dojo` contract.
"""

from pathlib import Path

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.generic.multi_card_option_selection.decoder_head import (
    MultiCardOptionSelectionDecoderHead,
)
from src.dojos.generic.option_scoring import OptionScoringHead
from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)
from src.dojos.loss.prior_baseline_calibrations import UniformOptionCalibration
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec
from src.schema.type_hints import TrainingInput


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
        deck_box: DeckBox | None = None,
        config: DojoConfig = DojoConfig(),
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
            calibration=UniformOptionCalibration(option_count=_option_count_of_pack),
            mod_pipeline=mod_pipeline,
            deck_box=deck_box,
            config=config,
        )


def _option_count_of_pack(pack_input: TrainingInput) -> int:
    """How many options a pack input offers: the input is the pack itself,
    one option per card (PackToPickChoiceSetDataConstructor).

    Inputs: pack_input, a MultiCardInput (list of option cards).
    Output: int, len(pack_input).
    Side effects: none. Exceptions: TypeError if pack_input is not a list.
    """
    if not isinstance(pack_input, list):
        raise TypeError(f"expected a pack (list of cards), got {type(pack_input)}")
    return len(pack_input)
