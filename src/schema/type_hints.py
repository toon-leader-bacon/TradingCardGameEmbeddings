"""Shared type aliases for card/embedding shapes used across dojos and models.

Every training input and model output in this project takes one of three
shapes - single card, multi card (an ordered list), or multi group (a list
of card lists) - per src/README.md's model taxonomy. These aliases name
that taxonomy once so dojos, encoder models, and training code all refer
to the same shapes instead of each re-deriving them.
"""

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
    MultiCardInput,   # List[GenericCard]
    MultiGroupInput   # List[List[GenericCard]]
]
BatchedTrainingInput = Union[
    BatchedSingleCardInput,  # List[GenericCard]
    BatchedMultiCardInput,   # List[List[GenericCard]]
    BatchedMultiGroupInput   # List[List[List[GenericCard]]]
]
# endregion Inputs

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

ModelOutput = Union[
    SingleCardEmbedding,
    MultiCardEmbedding,
    MultiGroupEmbedding
]
BatchedModelOutput = Union[
    BatchedSingleCardEmbedding,
    BatchedMultiCardEmbedding,
    BatchedMultiGroupEmbedding
]
# endregion Outputs
