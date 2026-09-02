from typing import List, Union

import torch
import torch.nn as nn

from src.schema.card import GenericCard
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


class MultiCardModel(nn.Module):
    def __init__(self, card_embedding_size: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.card_embedding_size = card_embedding_size

        # TODO: replace with the real embedder - a text serialization of
        # GenericCard.raw_content run through a text encoder. Placeholder
        # so self_attention below has something concrete to contextualize:
        # one learned vector per distinct card uuid, hashed into a fixed
        # vocab. Doesn't generalize to cards it hasn't seen during
        # training, unlike a real text encoder would.
        self.card_vocab_size = 100_000
        self.base_embedder = nn.Embedding(self.card_vocab_size, card_embedding_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_embedding_size,
            nhead=num_heads,
            batch_first=True,
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def internal_model(self, cards: List[GenericCard]) -> List[torch.Tensor]:
        """One contextualized embedding per input card, same order - every
        card attends to every other card in `cards` before its embedding
        is returned."""
        indices = torch.tensor([card.nocab_uuid.int % self.card_vocab_size for card in cards])
        base = self.base_embedder(indices)  # (num_cards, embedding_dim)

        # nn.TransformerEncoder expects a batch dim; treat this one group
        # of cards as its own batch of size 1, then drop it back off.
        contextualized = self.self_attention(base.unsqueeze(0)).squeeze(0)
        # (num_cards, embedding_dim) -> List[(embedding_dim,)]
        return list(contextualized.unbind(0))

    def forward(
        self,
        x: Union[SingleCardInput, MultiCardInput, MultiGroupInput, BatchedMultiGroupInput]
    ) -> Union[SingleCardEmbedding, MultiCardEmbedding, MultiGroupEmbedding, BatchedMultiGroupEmbedding]:
        if isinstance(x, SingleCardInput):
            return self.forward_single_card(x)
        elif isinstance(x, MultiCardInput) or isinstance(x, BatchedSingleCardInput):
            return self.forward_multi_card(x)
        elif isinstance(x, MultiGroupInput) or isinstance(x, BatchedMultiCardInput):
            return self.forward_multi_group(x)
        elif isinstance(x, BatchedMultiGroupInput):
            return self.forward_batched_multi_group(x)
        else:
            raise ValueError(f"Unsupported input type: {type(x)} for MultiCardModel ")

    # region Forward Methods

    def forward_single_card(self, x: SingleCardInput) -> SingleCardEmbedding:
        # Single card in, single embedding out
        # For this multi-card model, it only accepts a list of cards and outputs
        # a list of embeddings (one per card). So to support the single card case,
        # we wrap the single card in a list and pass it to the internal model,
        # then return the first (and only) embedding in the list
        model_out: List[torch.Tensor] = self.internal_model([x])
        return model_out[0]

    def forward_multi_card(self, x: Union[MultiCardInput, BatchedSingleCardInput]) -> Union[MultiCardEmbedding, BatchedSingleCardEmbedding]:
        return self.internal_model(x)

    def forward_multi_group(self, x: Union[MultiGroupInput, BatchedMultiCardInput]) -> Union[MultiGroupEmbedding, BatchedMultiCardEmbedding]:
        results: MultiGroupEmbedding = []
        for group in x:
            results.append(self.forward_multi_card(group))
        return results

    def forward_batched_multi_group(self, x: BatchedMultiGroupInput) -> BatchedMultiGroupEmbedding:
        results: BatchedMultiGroupEmbedding = []
        for multi_group in x:
            results.append(self.forward_multi_group(multi_group))
        return results

    # endregion Forward Methods
