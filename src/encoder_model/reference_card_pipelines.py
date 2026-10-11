"""Preset CardEmbeddingPipelines, shared by the single-card and multi-card presets
(reference_singlecard_models.py, reference_multicard_models.py), so each
pairing of text encoder, pooling and head is written once.

Factory functions, not classes: a preset is only a choice of stages.
Reference examples for demos, tests and scripts; a real experiment builds
its CardEmbeddingPipeline from its run config (scripts/run_training.py).
"""

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.embedding_head import (
    LinearEmbeddingHead,
    ResidualMlpEmbeddingHead,
)
from src.encoder_model.pooling.attention_pooling import AttentionPooling
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.encoder_model.text_encoder import (
    PretrainedTextEncoder,
    StaticEmbeddingTextEncoder,
)

# Cased and code-heavy-pretrained (card text is JSON), 8192-token native
# context (this project truncates well below that - see text_encoder.py's
# _DEFAULT_MAX_LENGTH) and Apache-2.0. No open checkpoint is JSON-tuned;
# this is the closest available - see src/training/Notes.md section 4 for
# the alternatives considered. Loads with attn_implementation="sdpa" by
# default (confirmed, no override needed).
DEFAULT_PRETRAINED_CHECKPOINT = "answerdotai/ModernBERT-base"


def build_linear_projection_pipeline(
    checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT, card_embedding_size: int = 256
) -> CardEmbeddingPipeline:
    """Floor baseline: frozen pretrained text encoder, mean pooling, one
    linear projection. Any richer combination should beat this.

    Inputs: checkpoint (Hugging Face id), card_embedding_size.
    Output: CardEmbeddingPipeline. Side effects: may download the checkpoint.
    Exceptions: whatever PretrainedTextEncoder raises.

    Example:
        >>> build_linear_projection_pipeline(card_embedding_size=256).output_dim
        256
    """
    text_encoder = PretrainedTextEncoder(checkpoint=checkpoint, trainable=False)
    pooling = MeanPooling(text_encoder.output_dim)
    head = LinearEmbeddingHead(pooling.output_dim, card_embedding_size)
    return CardEmbeddingPipeline(text_encoder, None, pooling, head)


def build_residual_mlp_pipeline(
    checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
    card_embedding_size: int = 256,
    hidden_dim: int = 512,
    num_blocks: int = 3,
) -> CardEmbeddingPipeline:
    """Frozen pretrained text encoder, mean pooling, residual-MLP head.

    Inputs: checkpoint, card_embedding_size, hidden_dim and num_blocks
        (the head's). Output: CardEmbeddingPipeline.
    Side effects: may download the checkpoint.
    Exceptions: whatever PretrainedTextEncoder raises.

    Example:
        >>> build_residual_mlp_pipeline(num_blocks=2).output_dim
        256
    """
    text_encoder = PretrainedTextEncoder(checkpoint=checkpoint, trainable=False)
    pooling = MeanPooling(text_encoder.output_dim)
    head = ResidualMlpEmbeddingHead(
        pooling.output_dim,
        card_embedding_size,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    )
    return CardEmbeddingPipeline(text_encoder, None, pooling, head)


def build_attention_pooling_pipeline(
    checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
    card_embedding_size: int = 256,
    num_queries: int = 1,
) -> CardEmbeddingPipeline:
    """Frozen pretrained text encoder, learned attention pooling, linear
    projection - instead of the other presets' mean pooling.

    Inputs: checkpoint, card_embedding_size, num_queries (the pooling's).
    Output: CardEmbeddingPipeline. Side effects: may download the checkpoint.
    Exceptions: whatever PretrainedTextEncoder or AttentionPooling raise.

    Example:
        >>> build_attention_pooling_pipeline(num_queries=4).pooling.output_dim
        3072
    """
    text_encoder = PretrainedTextEncoder(checkpoint=checkpoint, trainable=False)
    pooling = AttentionPooling(text_encoder.output_dim, num_queries=num_queries)
    head = LinearEmbeddingHead(pooling.output_dim, card_embedding_size)
    return CardEmbeddingPipeline(text_encoder, None, pooling, head)


def build_from_scratch_pipeline(
    tokenizer_checkpoint: str = DEFAULT_PRETRAINED_CHECKPOINT,
    token_embedding_dim: int = 256,
    card_embedding_size: int = 256,
) -> CardEmbeddingPipeline:
    """No pretrained weights: a trainable static token-embedding table,
    mean pooling and a residual-MLP head, all trained from the dojos.

    Inputs: tokenizer_checkpoint (subword splitting only),
        token_embedding_dim, card_embedding_size.
    Output: CardEmbeddingPipeline. Side effects: may download the tokenizer.
    Exceptions: whatever StaticEmbeddingTextEncoder raises.

    Example:
        >>> build_from_scratch_pipeline(token_embedding_dim=64).text_encoder.output_dim
        64
    """
    text_encoder = StaticEmbeddingTextEncoder(
        tokenizer_checkpoint=tokenizer_checkpoint, embedding_dim=token_embedding_dim
    )
    pooling = MeanPooling(text_encoder.output_dim)
    head = ResidualMlpEmbeddingHead(pooling.output_dim, card_embedding_size)
    return CardEmbeddingPipeline(text_encoder, None, pooling, head)
