from typing import List, Union, cast

import torch
import torch.nn as nn

from src.encoder_model.card_serialization import serialize_card_to_json
from src.encoder_model.embedding_head import EmbeddingHead
from src.encoder_model.text_encoder import TextEncoder
from src.schema.card import GenericCard
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
    def __init__(self, text_encoder: TextEncoder, embedding_head: EmbeddingHead):
        super().__init__()
        # Strategy + dependency injection: both collaborators are handed
        # in by the caller, so swapping the text encoder (pretrained
        # today, possibly a from-scratch-trained one later) or the head
        # architecture (linear, residual MLP, ...) never requires
        # touching this class.
        self.text_encoder: TextEncoder = text_encoder
        self.embedding_head: EmbeddingHead = embedding_head

    def internal_model(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One embedding per input card, same order - every card is
        serialized to text, batch-encoded, and pooled/projected by the
        injected TextEncoder/EmbeddingHead pair.

        Inputs: cards, a batch of GenericCard from any onboarded game.
        Output: list of embedding tensors, same length and order as cards.
        Side effects: none directly (the injected TextEncoder may
            accumulate gradients if it isn't frozen).
        Exceptions: none expected beyond whatever the injected
            TextEncoder/EmbeddingHead raise.
        """
        texts = [serialize_card_to_json(card) for card in cards]
        encoding = self.text_encoder.encode(texts)
        embeddings = self.embedding_head(encoding)
        return list(embeddings.unbind(0))

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
            return self.forward_single_card(cast(SingleCardInput, x))
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
    def forward_single_card(self, x: SingleCardInput) -> SingleCardEmbedding:
        # internal_model only accepts a list; wrap/unwrap for the
        # one-card case.
        return self.internal_model([x])[0]

    def forward_multi_card(
        self, x: Union[MultiCardInput, BatchedSingleCardInput]
    ) -> Union[MultiCardEmbedding, BatchedSingleCardEmbedding]:
        return self.internal_model(x)

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
