"""The encoder a training run builds, as frozen specs: ModelSpec and the
pooling, enrichment and group-attention specs inside it.

Parsed from a run config's `model:` section by
config_parsing/model_parsing.py; read by scripts/run_training.py when it
builds the CardEmbeddingPipeline. Sits beside plan.py, which holds the
rest of what a run config parses into.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class EmbeddingHeadKind(StrEnum):
    """Which EmbeddingHead (src/encoder_model/embedding_head.py) projects
    each card's pooled vector to its embedding."""

    LINEAR_PROJECTION = "linear_projection"
    RESIDUAL_MLP = "residual_mlp"


@dataclass(frozen=True)
class TokenWhiteningSpec:
    """`model.enrichment: {kind: whitening, stats_path: ...}`: whiten every
    token with the WhiteningStats at stats_path
    (src/encoder_model/enrichment/token_whitening.py; fitted by
    scripts/fit_whitening.py)."""

    stats_path: Path


@dataclass(frozen=True)
class MeanPoolingSpec:
    """`model.pooling: {kind: mean}`: the average of the real tokens."""


@dataclass(frozen=True)
class AttentionPoolingSpec:
    """`model.pooling: {kind: attention, num_queries: 4}`: learned queries,
    outputs concatenated (src/encoder_model/pooling/attention_pooling.py)."""

    num_queries: int = 1

    def __post_init__(self) -> None:
        """Exceptions: ValueError if num_queries < 1."""
        if self.num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {self.num_queries}")


@dataclass(frozen=True)
class TransformerPoolingSpec:
    """`model.pooling: {kind: transformer, num_layers: 1, inner: {kind:
    attention}}`: trainable transformer layers over the tokens, then the
    inner pooling (src/encoder_model/pooling/transformer_pooling.py).
    num_heads must divide the token width; that is checked when the model
    is built, since token whitening may change the width."""

    inner: MeanPoolingSpec | AttentionPoolingSpec = AttentionPoolingSpec()
    num_layers: int = 1
    num_heads: int = 8
    ffn_dim: int | None = None  # None: 2 * the token width
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """Exceptions: ValueError if num_layers, num_heads or ffn_dim (when
        given) < 1, or dropout outside [0, 1)."""
        sizes = {"num_layers": self.num_layers, "num_heads": self.num_heads}
        if self.ffn_dim is not None:
            sizes["ffn_dim"] = self.ffn_dim
        for name, size in sizes.items():
            if size < 1:
                raise ValueError(f"{name} must be >= 1, got {size}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


PoolingSpec = MeanPoolingSpec | AttentionPoolingSpec | TransformerPoolingSpec


@dataclass(frozen=True)
class GroupAttentionSpec:
    """The self-attention layers a multi-card model stacks on the
    embedding head (MultiCardModel), so cards in a group see each other.

    num_layers, num_heads: Transformer encoder layers and attention heads
        per layer (num_heads must divide card_embedding_size).
    ffn_dim: each layer's feed-forward width.
    dropout: dropout inside each layer.
    norm_first: pre-norm (LayerNorm before attention and feed-forward)
        rather than torch's default post-norm.
    """

    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    norm_first: bool = True

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_heads", "ffn_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class ModelSpec:
    """The encoder to build: a CardEmbeddingPipeline (frozen pretrained text
    encoder, optional token enrichment, pooling, embedding head), and for
    a multi-card model, group attention on top.

    embedding_head: which EmbeddingHead.
    checkpoint: Hugging Face checkpoint of the pretrained text encoder.
    card_embedding_size: output embedding width; every dojo head is built
        for it.
    mlp_hidden_dim, mlp_num_blocks: the residual-MLP head's width and
        block count (read only for EmbeddingHeadKind.RESIDUAL_MLP).
    group_attention: None for a single-card model (each card embedded
        alone); otherwise the MultiCardModel's self-attention layers.
    pooling: how each card's tokens become one vector (mean by default).
    enrichment: None, or token whitening before the pooling.
    """

    embedding_head: EmbeddingHeadKind
    checkpoint: str
    card_embedding_size: int
    mlp_hidden_dim: int = 512
    mlp_num_blocks: int = 3
    group_attention: GroupAttentionSpec | None = None
    # pooling and enrichment come last so older positional constructions
    # (ModelSpec(head, checkpoint, size, hidden, blocks)) keep their meaning
    pooling: PoolingSpec = MeanPoolingSpec()
    enrichment: TokenWhiteningSpec | None = None

    def __post_init__(self) -> None:
        if self.card_embedding_size < 1:
            raise ValueError("card_embedding_size must be >= 1")
        if self.mlp_hidden_dim < 1 or self.mlp_num_blocks < 0:
            raise ValueError("mlp_hidden_dim must be >= 1, mlp_num_blocks >= 0")
        attention = self.group_attention
        if attention is not None and self.card_embedding_size % attention.num_heads:
            raise ValueError(
                f"group_attention.num_heads ({attention.num_heads}) must divide "
                f"card_embedding_size ({self.card_embedding_size})"
            )
