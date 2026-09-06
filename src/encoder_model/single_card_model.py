from typing import Union, cast

import torch.nn as nn

from src.schema.type_hints import (
    BatchedMultiCardEmbedding,
    BatchedMultiCardInput,
    BatchedMultiGroupEmbedding,
    BatchedMultiGroupInput,
    BatchedSingleCardEmbedding,
    BatchedSingleCardInput,
    InputShape,
    MultiCardEmbedding,
    MultiCardInput,
    MultiGroupEmbedding,
    MultiGroupInput,
    SingleCardEmbedding,
    SingleCardInput,
    input_shape_of,
)


class SingleCardModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Internal model is expected to take in a GenericCard and return a
        # single embedding tensor
        self.internal_model: nn.Module = None  # TODO: Implement this

    def forward(
        self,
        x: Union[
            SingleCardInput, MultiCardInput, MultiGroupInput, BatchedMultiGroupInput
        ],
    ) -> Union[
        SingleCardEmbedding,
        MultiCardEmbedding,
        MultiGroupEmbedding,
        BatchedMultiGroupEmbedding,
    ]:
        shape = input_shape_of(x)
        if shape is InputShape.SINGLE_CARD:
            # Simple case, one card in one embedding out
            return self.internal_model(cast(SingleCardInput, x).embedding)
        elif shape is InputShape.MULTI_CARD:
            # Batch of single cards, or a single multi-card input - same
            # runtime shape (one list of GenericCard). One embedding per
            # card, organized in the same order as the input list.
            return self.forward_multi_card(
                cast(Union[MultiCardInput, BatchedSingleCardInput], x)
            )
        elif shape is InputShape.MULTI_GROUP:
            # Batch of multi-card inputs, or a single multi-group input -
            # same runtime shape. Still one embedding per card, still
            # organized in the same order as the input list of lists.
            return self.forward_multi_group(
                cast(Union[MultiGroupInput, BatchedMultiCardInput], x)
            )
        else:
            # Batch of multi-group inputs.
            return self.forward_batched_multi_group(cast(BatchedMultiGroupInput, x))

    # region Forward Methods
    def forward_multi_card(
        self, x: Union[MultiCardInput, BatchedSingleCardInput]
    ) -> Union[MultiCardEmbedding, BatchedSingleCardEmbedding]:
        results: MultiCardEmbedding = []
        for card in x:
            results.append(self.internal_model(card.embedding))
        return results

    def forward_multi_group(
        self, x: Union[MultiGroupInput, BatchedMultiCardInput]
    ) -> Union[MultiGroupEmbedding, BatchedMultiCardEmbedding]:
        results: MultiGroupEmbedding = []
        for group in x:
            results.append(self.forward_multi_card(group))
        return results

    def forward_batched_multi_group(
        self, x: BatchedMultiGroupInput
    ) -> BatchedMultiGroupEmbedding:
        results: BatchedMultiGroupEmbedding = []
        for multi_group in x:
            results.append(self.forward_multi_group(multi_group))
        return results

    # endregion Forward Methods
