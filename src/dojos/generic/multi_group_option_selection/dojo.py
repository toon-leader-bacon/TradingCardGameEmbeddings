"""Generic dojo for the (pack of options + a conditioning group in, picked option out) task shape.

A GenericDojo (src/dojos/generic/generic_dojo.py) that supplies this cell's
decoder head and loss; everything else is the shared `Dojo` contract.
"""

from pathlib import Path
from typing import Callable

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.generic.data_constructor import DataConstructor
from src.dojos.generic.dojo_config import DojoConfig
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.generic.multi_group_option_selection.decoder_head import (
    MultiGroupOptionSelectionDecoderHead,
)
from src.dojos.generic.option_scoring import OptionScoringHead
from src.dojos.generic.pooling import EmbeddingPooler
from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)
from src.dojos.loss.prior_baseline_calibrations import UniformOptionCalibration
from src.dojos.mods.mod_pipeline import ModPipeline
from src.schema.holdout import HoldoutSpec
from src.schema.type_hints import TrainingInput


class MultiGroupOptionSelectionDojo(GenericDojo):
    """Dojo for the (pack of options + a conditioning group in, picked option out) task shape.

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
        pooler: EmbeddingPooler | None = None,
        deck_box: DeckBox | None = None,
        config: DojoConfig = DojoConfig(),
        can_skip: bool = False,
    ) -> None:
        """
        Inputs: as GenericDojo, plus can_skip: True adds a learned "pick
            none of them" option after each example's options (see
            MultiGroupOptionSelectionDecoderHead); the data constructor
            then labels a skip as len(options), and the loss baseline
            counts that extra option. False (default): picks only.
        Output: none (constructor).
        Side effects, Exceptions: as GenericDojo.
        """
        self.card_embedding_size = card_embedding_size
        super().__init__(
            path_to_training_data=path_to_training_data,
            data_constructor=data_constructor,
            card_lookup=card_lookup,
            holdout=holdout,
            decoder_head=MultiGroupOptionSelectionDecoderHead(
                card_embedding_size,
                scoring_head=scoring_head,
                pooler=pooler,
                learns_skip_option=can_skip,
            ),
            loss_calculator=PickPredictionCrossEntropyLoss(),
            calibration=UniformOptionCalibration(
                option_count=_option_count_fn(can_skip)
            ),
            mod_pipeline=mod_pipeline,
            deck_box=deck_box,
            config=config,
        )


def _option_count_fn(can_skip: bool) -> Callable[[TrainingInput], int]:
    """The input -> option count function for the loss baseline: the
    pack's size, plus one when the cell has a skip option.

    Inputs: can_skip. Output: a function of a TrainingInput.
    Side effects: none. Exceptions: none (the function it returns raises
        as _option_count_of_pack_group does).
    """
    return _option_count_with_skip if can_skip else _option_count_of_pack_group


def _option_count_with_skip(group_input: TrainingInput) -> int:
    """The pack's size plus the skip option.

    Inputs: group_input, as _option_count_of_pack_group.
    Output: int. Side effects: none.
    Exceptions: as _option_count_of_pack_group.
    """
    return _option_count_of_pack_group(group_input) + 1


def _option_count_of_pack_group(group_input: TrainingInput) -> int:
    """How many options a (pack, conditioning group) input offers: the
    size of group 0, the pack (every constructor feeding this cell, e.g.
    PoolConditionedPickDataConstructor and CardRewardPickDataConstructor,
    keeps the options at index 0 and the conditioning group at index 1).

    Inputs: group_input, a MultiGroupInput [pack_cards, pool_cards].
    Output: int, len(pack_cards).
    Side effects: none. Exceptions: TypeError if group_input is not a
        list of card lists.
    """
    if not isinstance(group_input, list) or not group_input:
        raise TypeError(f"expected [pack_cards, pool_cards], got {type(group_input)}")
    pack = group_input[0]
    if not isinstance(pack, list):
        raise TypeError(f"expected the pack (group 0) to be a list, got {type(pack)}")
    return len(pack)
