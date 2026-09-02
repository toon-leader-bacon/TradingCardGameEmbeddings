from typing import Union

import torch.nn as nn

from src.schema.type_hints import (
    BatchedMultiCardEmbedding,
    BatchedMultiCardInput,
    BatchedMultiGroupEmbedding,
    BatchedMultiGroupInput,
    BatchedSingleCardEmbedding,
    BatchedSingleCardInput,
    MultiCardEmbedding,
    MultiCardInput,
    MultiGroupEmbedding,
    MultiGroupInput,
    SingleCardEmbedding,
    SingleCardInput,
)


class SingleCardModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Internal model is expected to take in a GenericCard and return a
        # single embedding tensor
        self.internal_model: nn.Module = None  # TODO: Implement this

    def forward(
        self,
        x: Union[SingleCardInput, MultiCardInput, MultiGroupInput, BatchedMultiGroupInput]
    ) -> Union[SingleCardEmbedding, MultiCardEmbedding, MultiGroupEmbedding, BatchedMultiGroupEmbedding]:
        if isinstance(x, SingleCardInput):
            # Simple case, one card in one embedding out
            return self.internal_model(x.embedding)
        elif isinstance(x, MultiCardInput) or isinstance(x, BatchedSingleCardInput):
            # Batch of single cards, or a single multi-card input
            # Multi-card case. One embedding per card, organized in the same
            # order as the input list
            return self.forward_multi_card(x)
        elif isinstance(x, MultiGroupInput) or isinstance(x, BatchedMultiCardInput):
            # Batch of multi-card inputs, or a single multi-group input
            # Multi-group case. Still one embedding per card, still organized
            # in the same order as the input list of lists
            return self.forward_multi_group(x)
        elif isinstance(x, BatchedMultiGroupInput):
            # Batch of multi-group inputs, or a single multi-group input
            return self.forward_batched_multi_group(x)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

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
