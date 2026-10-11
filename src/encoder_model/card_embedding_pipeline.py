"""CardEmbeddingPipeline: cards -> one embedding each, through four injected stages.

text encoder -> optional token enrichment -> token pooling -> embedding
head. SingleCardModel and MultiCardModel each own one; this is the only
place the stages are chained. See README.md.
"""

from typing import List

import torch
import torch.nn as nn

from src.encoder_model.card_serialization import serialize_card_to_json
from src.encoder_model.embedding_head import EmbeddingHead
from src.encoder_model.enrichment.token_enrichment import TokenEnrichment
from src.encoder_model.pooling.token_pooling import TokenPooling
from src.encoder_model.text_encoder import TextEncoder
from src.schema.card import GenericCard


class CardEmbeddingPipeline(nn.Module):
    """Every card on its own: serialized to text, encoded to token states,
    optionally enriched, pooled to one vector, projected by the head. No
    card sees another.

    Strategy + dependency injection: all four stages are handed in, so any
    one can be swapped without touching this class or the models.
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        enrichment: TokenEnrichment | None,
        pooling: TokenPooling,
        embedding_head: EmbeddingHead,
    ) -> None:
        """Inputs: text_encoder, enrichment (None: tokens go straight to
            the pooling), pooling, embedding_head.
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if adjacent widths do not chain:
            text_encoder.output_dim -> enrichment -> pooling.input_dim, and
            pooling.output_dim -> embedding_head.input_dim.

        Example:
            >>> encoder = PretrainedTextEncoder("answerdotai/ModernBERT-base")
            >>> pipeline = CardEmbeddingPipeline(encoder, None, MeanPooling(768),
            ...                         LinearEmbeddingHead(768, 256))
            >>> pipeline.output_dim
            256
        """
        super().__init__()
        # Validate that each stage's output width is the next one's input
        _check_widths_chain(text_encoder, enrichment, pooling, embedding_head)
        self.text_encoder: TextEncoder = text_encoder
        self.enrichment: TokenEnrichment | None = enrichment
        self.pooling: TokenPooling = pooling
        self.embedding_head: EmbeddingHead = embedding_head

    @property
    def output_dim(self) -> int:
        """The embedding width: the head's output_dim. Inputs: none.
        Output: int. Side effects: none. Exceptions: none."""
        return self.embedding_head.output_dim

    def embed(self, cards: List[GenericCard]) -> torch.Tensor:
        """One embedding per card, same order, each independent of the
        others.

        Inputs: cards (List[GenericCard]), non-empty, from any game.
        Output: (len(cards), output_dim) tensor.
        Side effects: none directly (a trainable text encoder accumulates
            gradients as a normal side effect of autograd).
        Exceptions: whatever the injected stages raise.

        Example:
            >>> pipeline.embed([card_a, card_b]).shape
            torch.Size([2, 256])
        """
        # Serialize and encode every card in one batch
        texts = [serialize_card_to_json(card) for card in cards]
        encoding = self.text_encoder.encode(texts)
        # Enrich the tokens, when this pipeline has an enrichment
        if self.enrichment is not None:
            encoding = self.enrichment(encoding)
        # Pool each card's tokens, then project
        pooled = self.pooling(encoding)
        return self.embedding_head(pooled)


def _check_widths_chain(
    text_encoder: TextEncoder,
    enrichment: TokenEnrichment | None,
    pooling: TokenPooling,
    embedding_head: EmbeddingHead,
) -> None:
    """Private helper - single consumer is CardEmbeddingPipeline.__init__.

    Inputs: the four stages, as CardEmbeddingPipeline.__init__.
    Output: none. Side effects: none.
    Exceptions: ValueError naming the first pair of stages whose widths
        differ (e.g. "text encoder output_dim 768 != pooling input_dim 256").
    """
    # Each (producer name, width out, consumer name, width in), in order
    links: list[tuple[str, int, str, int]] = []
    token_source, token_width = "text encoder", text_encoder.output_dim
    if enrichment is not None:
        links.append((token_source, token_width, "enrichment", enrichment.input_dim))
        token_source, token_width = "enrichment", enrichment.output_dim
    links.append((token_source, token_width, "pooling", pooling.input_dim))
    links.append(
        ("pooling", pooling.output_dim, "embedding head", embedding_head.input_dim)
    )

    for producer, width_out, consumer, width_in in links:
        if width_out != width_in:
            raise ValueError(
                f"{producer} output_dim {width_out} != {consumer} input_dim {width_in}"
            )
