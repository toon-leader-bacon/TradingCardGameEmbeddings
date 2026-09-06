"""Shared type aliases for card/embedding shapes used across dojos and models.

Every training input and model output in this project takes one of three
shapes - single card, multi card (an ordered list), or multi group (a list
of card lists) - per src/README.md's model taxonomy. These aliases name
that taxonomy once so dojos, encoder models, and training code all refer
to the same shapes instead of each re-deriving them.
"""

from enum import Enum
from typing import Any, List, Tuple, Union

import torch

from src.schema.card import GenericCard

# region Inputs
SingleCardInput = GenericCard
BatchedSingleCardInput = List[SingleCardInput]

MultiCardInput = List[GenericCard]
BatchedMultiCardInput = List[MultiCardInput]

MultiGroupInput = List[MultiCardInput]
BatchedMultiGroupInput = List[MultiGroupInput]

TrainingInput = Union[
    SingleCardInput,  # GenericCard
    MultiCardInput,  # List[GenericCard]
    MultiGroupInput,  # List[List[GenericCard]]
]
BatchedTrainingInput = Union[
    BatchedSingleCardInput,  # List[GenericCard]
    BatchedMultiCardInput,  # List[List[GenericCard]]
    BatchedMultiGroupInput,  # List[List[List[GenericCard]]]
]
# endregion Inputs


class InputShape(Enum):
    """Which of the nested-list-of-GenericCard shapes above a value has.

    SingleCardInput/MultiCardInput/MultiGroupInput (and their Batched*
    counterparts one list-level up) are all just plain `list`/`GenericCard`
    at runtime - the type parameters are erased, so `isinstance(x,
    MultiCardInput)` isn't just unhelpful, it's a TypeError (Python refuses
    isinstance against a subscripted generic). Nesting depth is the only
    thing that actually distinguishes these shapes at runtime, so
    `input_shape_of` counts it instead.
    """

    SINGLE_CARD = 0
    MULTI_CARD = 1  # also BatchedSingleCardInput - same runtime shape
    MULTI_GROUP = 2  # also BatchedMultiCardInput - same runtime shape
    BATCHED_MULTI_GROUP = 3


def input_shape_of(
    x: Union[TrainingInput, BatchedMultiGroupInput],
) -> InputShape:
    """Classify a value by GenericCard-list nesting depth.

    Inputs: x, a GenericCard optionally nested in 0-3 levels of list.
    Output: the matching InputShape.
    Side effects: none.
    Exceptions: ValueError if x bottoms out in something other than a
        GenericCard, or an empty list is encountered before reaching one
        (depth is then ambiguous).

    Example:
        >>> input_shape_of([some_card, other_card])
        <InputShape.MULTI_CARD: 1>
    """
    depth = 0
    probe: Any = x
    while isinstance(probe, list):
        if not probe:
            raise ValueError("Cannot classify the shape of an empty list input")
        probe = probe[0]
        depth += 1

    if not isinstance(probe, GenericCard):
        raise ValueError(f"Not a valid TrainingInput shape: {x}")
    return InputShape(depth)


Label = Any  # Typically a single scaler value, but could be a list of values
TrainingDatum = Tuple[TrainingInput, Label]


# region Outputs
Embedding = torch.Tensor

SingleCardEmbedding = Embedding
BatchedSingleCardEmbedding = List[SingleCardEmbedding]

MultiCardEmbedding = List[Embedding]
BatchedMultiCardEmbedding = List[MultiCardEmbedding]

MultiGroupEmbedding = List[MultiCardEmbedding]
BatchedMultiGroupEmbedding = List[MultiGroupEmbedding]

ModelOutput = Union[SingleCardEmbedding, MultiCardEmbedding, MultiGroupEmbedding]
BatchedModelOutput = Union[
    BatchedSingleCardEmbedding, BatchedMultiCardEmbedding, BatchedMultiGroupEmbedding
]
# endregion Outputs
