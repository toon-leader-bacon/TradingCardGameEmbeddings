"""A handful of concrete MultiCardModel presets, each a MultiCardModel
around one preset CardEmbeddingPipeline (reference_card_pipelines.py) plus the
self-attention sizing - mirrors reference_singlecard_models.py.

These are reference examples, not the primary way to construct a
MultiCardModel - a real experiment injects its own CardEmbeddingPipeline, not a
new preset class per combination tried.
"""

from src.encoder_model.multi_card_model import MultiCardModel
from src.encoder_model.reference_card_pipelines import (
    DEFAULT_PRETRAINED_CHECKPOINT,
    build_attention_pooling_pipeline,
    build_from_scratch_pipeline,
    build_linear_projection_pipeline,
    build_residual_mlp_pipeline,
)


class LinearProjectionMultiCardModel(MultiCardModel):
    """Floor baseline: frozen text encoder, mean pooling, linear head,
    feeding the self-attention group encoder."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
        card_embedding_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """Inputs: see build_linear_projection_pipeline;
            num_heads and num_layers size the group self-attention.
        Output: none (constructor). Side effects: may download the
        checkpoint. Exceptions: as build_linear_projection_pipeline."""
        super().__init__(
            build_linear_projection_pipeline(checkpoint, card_embedding_size),
            num_heads=num_heads,
            num_layers=num_layers,
        )


class ResidualMlpMultiCardModel(MultiCardModel):
    """Frozen text encoder, mean pooling, residual-MLP head, feeding the
    self-attention group encoder."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
        card_embedding_size: int = 256,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """Inputs: see build_residual_mlp_pipeline;
            num_heads and num_layers size the group self-attention.
        Output: none (constructor). Side effects: may download the
        checkpoint. Exceptions: as build_residual_mlp_pipeline."""
        super().__init__(
            build_residual_mlp_pipeline(
                checkpoint, card_embedding_size, hidden_dim, num_blocks
            ),
            num_heads=num_heads,
            num_layers=num_layers,
        )


class AttentionPoolingMultiCardModel(MultiCardModel):
    """Frozen text encoder, learned attention pooling, linear head,
    feeding the self-attention group encoder."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
        card_embedding_size: int = 256,
        num_queries: int = 1,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """Inputs: see build_attention_pooling_pipeline;
            num_heads and num_layers size the group self-attention.
        Output: none (constructor). Side effects: may download the
        checkpoint. Exceptions: as build_attention_pooling_pipeline."""
        super().__init__(
            build_attention_pooling_pipeline(
                checkpoint, card_embedding_size, num_queries
            ),
            num_heads=num_heads,
            num_layers=num_layers,
        )


class FromScratchMultiCardModel(MultiCardModel):
    """No pretrained weights anywhere: a trainable static token-embedding
    table, mean pooling and a residual-MLP head, feeding the
    self-attention group encoder."""

    def __init__(
        self,
        tokenizer_checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
        token_embedding_dim: int = 256,
        card_embedding_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """Inputs: see build_from_scratch_pipeline;
            num_heads and num_layers size the group self-attention.
        Output: none (constructor). Side effects: may download the
        checkpoint. Exceptions: as build_from_scratch_pipeline."""
        super().__init__(
            build_from_scratch_pipeline(
                tokenizer_checkpoint, token_embedding_dim, card_embedding_size
            ),
            num_heads=num_heads,
            num_layers=num_layers,
        )
