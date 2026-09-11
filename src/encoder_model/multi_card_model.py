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


class MultiCardModel(nn.Module):
    def __init__(
        self,
        text_encoder: TextEncoder,
        embedding_head: EmbeddingHead,
        card_embedding_size: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        # Strategy + dependency injection, same seam as SingleCardModel:
        # text_encoder/embedding_head turn each card into a base embedding
        # (card_embedding_size wide) before self_attention below lets every
        # card in the group attend to every other card.
        self.text_encoder: TextEncoder = text_encoder
        self.embedding_head: EmbeddingHead = embedding_head
        self.card_embedding_size = card_embedding_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_embedding_size,
            nhead=num_heads,
            batch_first=True,
        )
        self.self_attention = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def internal_model(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One contextualized embedding per input card, same order - every
        card attends to every other card in `cards` before its embedding
        is returned."""
        texts = [serialize_card_to_json(card) for card in cards]
        encoding = self.text_encoder.encode(texts)
        base = self.embedding_head(encoding)  # (num_cards, card_embedding_size)

        # nn.TransformerEncoder expects a batch dim; treat this one group
        # of cards as its own batch of size 1, then drop it back off.
        contextualized = self.self_attention(base.unsqueeze(0)).squeeze(0)
        # (num_cards, embedding_dim) -> List[(embedding_dim,)]
        return list(contextualized.unbind(0))

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
            return self.forward_single_card(cast(SingleCardInput, x))
        elif shape is InputShape.MULTI_CARD:
            return self.forward_multi_card(
                cast(Union[MultiCardInput, BatchedSingleCardInput], x)
            )
        elif shape is InputShape.MULTI_GROUP:
            return self.forward_multi_group(
                cast(Union[MultiGroupInput, BatchedMultiCardInput], x)
            )
        else:
            return self.forward_batched_multi_group(cast(BatchedMultiGroupInput, x))

    # region Forward Methods

    def forward_single_card(self, x: SingleCardInput) -> SingleCardEmbedding:
        # Single card in, single embedding out
        # For this multi-card model, it only accepts a list of cards and outputs
        # a list of embeddings (one per card). So to support the single card case,
        # we wrap the single card in a list and pass it to the internal model,
        # then return the first (and only) embedding in the list
        model_out: List[torch.Tensor] = self.internal_model([x])
        return model_out[0]

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
