"""The encoder a training run builds, as frozen specs: ModelSpec and the
pooling, enrichment and group-attention specs inside it.

Parsed from a run config's `model:` section by
config_parsing/model_parsing.py; read by scripts/run_training.py when it
builds the CardEmbeddingPipeline. Sits beside plan.py, which holds the
rest of what a run config parses into.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype


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

    KIND: ClassVar[str] = "mean"


@dataclass(frozen=True)
class AttentionPoolingSpec:
    """`model.pooling: {kind: attention, num_queries: 4}`: learned queries,
    outputs concatenated (src/encoder_model/pooling/attention_pooling.py)."""

    KIND: ClassVar[str] = "attention"

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

    KIND: ClassVar[str] = "transformer"

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


_POOLING_SPECS = (MeanPoolingSpec, AttentionPoolingSpec, TransformerPoolingSpec)


def pooling_label(spec: PoolingSpec) -> str:
    """A stable label for a pooling: its config kind, plus its fields as
    sorted JSON when it has any (part of a feature cache's fingerprint, so
    renaming a class never orphans cached rows).

    A nested pooling spec (a transformer's inner pooling) is labelled
    recursively, kind included, so two kinds with no fields never share a
    label.

    Inputs: spec. Output: str, e.g. "mean" or
        'attention{"num_queries":4}'. Side effects: none.
    Exceptions: none.

    Example:
        >>> pooling_label(MeanPoolingSpec())
        'mean'
    """
    fields = {
        name: pooling_label(value) if isinstance(value, _POOLING_SPECS) else value
        for name, value in vars(spec).items()
    }
    if not fields:
        return spec.KIND
    return spec.KIND + json.dumps(fields, sort_keys=True, separators=(",", ":"))


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


class FeatureCut(StrEnum):
    """Where a feature cache cuts the card pipeline: everything before the
    cut must be frozen, and its output is what the cache stores."""

    POOLED = "pooled"  # after the pooling: one vector per card


class SlotPolicyKind(StrEnum):
    """Which SlotPolicy a memory tier uses
    (src/encoder_model/feature_cache/slot_policy.py)."""

    LRU = "lru"  # evict the least recently used to admit new rows
    FIFO = "fifo"  # evict the oldest admitted; hits change nothing
    LFU = "lfu"  # evict the least hit, oldest first among ties
    FILL_UNTIL_FULL = "fill_until_full"  # admit while there is room, never evict


class AdmissionKind(StrEnum):
    """When a memory tier's policy is first asked to admit a key."""

    FIRST_OFFER = "first_offer"  # every offer reaches the policy
    SECOND_OFFER = "second_offer"  # a key's first offer only is remembered


class DiskWrite(StrEnum):
    """Which encodes the disk tier keeps."""

    WARM_ONLY = "warm_only"  # only warm()/warm_feature_cache.py
    ALL = "all"  # also every cacheable training miss (random mods' cards never)


@dataclass(frozen=True)
class MemoryTierSpec:
    """`{capacity_mib: 512, policy: lru, admission: first_offer}`: a VRAM
    or pinned-RAM slab."""

    capacity_mib: int
    policy: SlotPolicyKind = SlotPolicyKind.LRU
    admission: AdmissionKind = AdmissionKind.FIRST_OFFER

    def __post_init__(self) -> None:
        """Inputs: none (checks self). Output: none. Side effects: none.
        Exceptions: ValueError if capacity_mib < 1."""
        if self.capacity_mib < 1:
            raise ValueError(f"capacity_mib must be >= 1, got {self.capacity_mib}")


@dataclass(frozen=True)
class DiskTierSpec:
    """`{path: data/feature_cache/features.sqlite, capacity_mib: 20480,
    write: warm_only}`: the SQLite tier; capacity counts row payload for
    this run's fingerprint."""

    path: Path
    capacity_mib: int
    write: DiskWrite = DiskWrite.WARM_ONLY

    def __post_init__(self) -> None:
        """Inputs: none (checks self). Output: none. Side effects: none.
        Exceptions: ValueError if capacity_mib < 1."""
        if self.capacity_mib < 1:
            raise ValueError(f"capacity_mib must be >= 1, got {self.capacity_mib}")


@dataclass(frozen=True)
class FeatureCacheSpec:
    """`model.feature_cache: {cut: pooled, storage_dtype: fp32, vram: {...},
    ram: {...}, disk: {...}}`: cache the frozen stages' output in a chain
    of tiers, fastest first (src/encoder_model/feature_cache/).

    cut: where the pipeline is cut (only POOLED today).
    storage_dtype: how every tier stores rows (results are always float32).
    vram, ram, disk: each tier, or None for none; at least one is set.
    """

    cut: FeatureCut = FeatureCut.POOLED
    storage_dtype: FeatureStorageDtype = FeatureStorageDtype.FP32
    vram: MemoryTierSpec | None = MemoryTierSpec(512)
    ram: MemoryTierSpec | None = None
    disk: DiskTierSpec | None = None

    def __post_init__(self) -> None:
        """Inputs: none (checks self). Output: none. Side effects: none.
        Exceptions: ValueError if every tier is None."""
        if self.vram is None and self.ram is None and self.disk is None:
            raise ValueError("feature_cache needs at least one of vram, ram, disk")


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
    feature_cache: None (every stage runs on every card), or a cache of
        the frozen stages' output; a pooled cut needs mean pooling.
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
    feature_cache: FeatureCacheSpec | None = None

    def __post_init__(self) -> None:
        """Exceptions: ValueError for an invalid size, group-attention heads
        that do not divide card_embedding_size, or a feature cache cut
        after a trainable stage."""
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
        cache = self.feature_cache
        if (
            cache is not None
            and cache.cut is FeatureCut.POOLED
            and not isinstance(self.pooling, MeanPoolingSpec)
        ):
            raise ValueError(
                "feature_cache cut 'pooled' needs mean pooling: "
                f"{type(self.pooling).__name__} is trainable"
            )
