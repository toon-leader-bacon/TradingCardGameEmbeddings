"""A handful of concrete SingleCardModel presets, each a SingleCardModel
around one preset CardEmbeddingPipeline (reference_card_pipelines.py).

These are reference examples, not the primary way to construct a
SingleCardModel - a real experiment injects its own CardEmbeddingPipeline, not a
new preset class per combination tried.
"""

from src.encoder_model.reference_card_pipelines import (
    DEFAULT_PRETRAINED_CHECKPOINT,
    build_attention_pooling_pipeline,
    build_from_scratch_pipeline,
    build_linear_projection_pipeline,
    build_residual_mlp_pipeline,
)
from src.encoder_model.single_card_model import SingleCardModel


class LinearProjectionCardModel(SingleCardModel):
    """Floor baseline: frozen text encoder, mean pooling, linear head."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
        card_embedding_size: int = 256,
    ):
        """Inputs: see build_linear_projection_pipeline.
        Output: none (constructor). Side effects: may download the
        checkpoint. Exceptions: as build_linear_projection_pipeline."""
        super().__init__(
            build_linear_projection_pipeline(checkpoint, card_embedding_size)
        )


class ResidualMlpCardModel(SingleCardModel):
    """Frozen text encoder, mean pooling, residual-MLP head."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
        card_embedding_size: int = 256,
        hidden_dim: int = 512,
        num_blocks: int = 3,
    ):
        """Inputs: see build_residual_mlp_pipeline.
        Output: none (constructor). Side effects: may download the
        checkpoint. Exceptions: as build_residual_mlp_pipeline."""
        super().__init__(
            build_residual_mlp_pipeline(
                checkpoint, card_embedding_size, hidden_dim, num_blocks
            )
        )


class AttentionPoolingCardModel(SingleCardModel):
    """Frozen text encoder, learned attention pooling, linear head."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
        card_embedding_size: int = 256,
        num_queries: int = 1,
    ):
        """Inputs: see build_attention_pooling_pipeline.
        Output: none (constructor). Side effects: may download the
        checkpoint. Exceptions: as build_attention_pooling_pipeline."""
        super().__init__(
            build_attention_pooling_pipeline(
                checkpoint, card_embedding_size, num_queries
            )
        )


class FromScratchCardModel(SingleCardModel):
    """No pretrained weights anywhere: a trainable static token-embedding
    table, mean pooling and a residual-MLP head."""

    def __init__(
        self,
        tokenizer_checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
        token_embedding_dim: int = 256,
        card_embedding_size: int = 256,
    ):
        """Inputs: see build_from_scratch_pipeline.
        Output: none (constructor). Side effects: may download the
        checkpoint. Exceptions: as build_from_scratch_pipeline."""
        super().__init__(
            build_from_scratch_pipeline(
                tokenizer_checkpoint, token_embedding_dim, card_embedding_size
            )
        )
