"""A run config's `model:` section -> ModelSpec (../model_spec.py)."""

from pathlib import Path
from typing import Callable

from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.training.config_parsing.config_section import (
    ConfigSection,
    construct_at,
    parse_by_kind,
)
from src.training.model_spec import (
    AttentionPoolingSpec,
    DiskTierSpec,
    DiskWrite,
    EmbeddingHeadKind,
    FeatureCacheSpec,
    FeatureCut,
    GroupAttentionSpec,
    MeanPoolingSpec,
    AdmissionKind,
    MemoryTierSpec,
    ModelSpec,
    PoolingSpec,
    SlotPolicyKind,
    TokenWhiteningSpec,
    TransformerPoolingSpec,
)


def parse_model(section: ConfigSection) -> ModelSpec:
    """{"embedding_head", "checkpoint", "card_embedding_size",
    ["mlp_hidden_dim", "mlp_num_blocks"], ["group_attention"], ["pooling"],
    ["enrichment"], ["feature_cache"]} -> ModelSpec.

    Inputs: section ("model"). Output: ModelSpec.
    Side effects: none.
    Exceptions: ValueError for an unknown embedding_head, mlp_* keys on a
        head other than residual_mlp, an unknown pooling or enrichment
        kind, an invalid size, or a feature cache that is malformed or cut
        after a trainable pooling.
    """
    # Which head; the mlp_* sizes only mean something for the residual MLP
    head_text = section.required("embedding_head", str)
    if head_text not in {kind.value for kind in EmbeddingHeadKind}:
        choices = [kind.value for kind in EmbeddingHeadKind]
        raise ValueError(
            f"{section.location_of('embedding_head')} is {head_text!r}, "
            f"want one of {choices}"
        )
    head = EmbeddingHeadKind(head_text)
    mlp_keys = [key for key in ("mlp_hidden_dim", "mlp_num_blocks") if section.has(key)]
    if mlp_keys and head is not EmbeddingHeadKind.RESIDUAL_MLP:
        raise ValueError(
            f"{section.where}: {mlp_keys} apply only to embedding_head: residual_mlp"
        )

    # Sizes, then group attention (absent = a single-card model)
    checkpoint = section.required("checkpoint", str)
    card_embedding_size = section.required("card_embedding_size", int)
    mlp_hidden_dim = section.optional("mlp_hidden_dim", int, ModelSpec.mlp_hidden_dim)
    mlp_num_blocks = section.optional("mlp_num_blocks", int, ModelSpec.mlp_num_blocks)
    group_attention = None
    if section.has("group_attention"):
        group_attention = _parse_group_attention(
            section.required_section("group_attention")
        )

    # Pooling (mean when absent) and the optional token enrichment
    pooling: PoolingSpec = MeanPoolingSpec()
    if section.has("pooling"):
        pooling = parse_by_kind(section.required_section("pooling"), _POOLING_PARSERS)
    enrichment = None
    if section.has("enrichment"):
        enrichment = parse_by_kind(
            section.required_section("enrichment"), _ENRICHMENT_PARSERS
        )
    # The optional cache of the frozen stages' output
    feature_cache = None
    if section.has("feature_cache"):
        feature_cache = _parse_feature_cache(section.required_section("feature_cache"))
    section.reject_unread_keys()
    return construct_at(
        section.where,
        lambda: ModelSpec(
            embedding_head=head,
            checkpoint=checkpoint,
            card_embedding_size=card_embedding_size,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_blocks=mlp_num_blocks,
            group_attention=group_attention,
            pooling=pooling,
            enrichment=enrichment,
            feature_cache=feature_cache,
        ),
    )


def _parse_transformer_pooling(section: ConfigSection) -> TransformerPoolingSpec:
    """`{kind: transformer, [inner], [num_layers], [num_heads], [ffn_dim],
    [dropout]}` -> TransformerPoolingSpec; inner is a mean or attention
    pooling section (attention when absent), never another transformer.

    Private helper - single consumer is _POOLING_PARSERS.
    Inputs: section. Output: TransformerPoolingSpec. Side effects: none.
    Exceptions: ValueError for a nested transformer inner or a bad value.
    """
    defaults = TransformerPoolingSpec()
    inner = defaults.inner
    if section.has("inner"):
        parsed = parse_by_kind(section.required_section("inner"), _POOLING_PARSERS)
        if not isinstance(parsed, (MeanPoolingSpec, AttentionPoolingSpec)):
            raise ValueError(
                f"{section.location_of('inner')} must be a mean or attention pooling"
            )
        inner = parsed
    ffn_dim = section.required("ffn_dim", int) if section.has("ffn_dim") else None
    return TransformerPoolingSpec(
        inner=inner,
        num_layers=section.optional("num_layers", int, defaults.num_layers),
        num_heads=section.optional("num_heads", int, defaults.num_heads),
        ffn_dim=ffn_dim,
        dropout=section.optional("dropout", float, defaults.dropout),
    )


# Each pooling kind's parser, by its config name
_POOLING_PARSERS: dict[str, Callable[[ConfigSection], PoolingSpec]] = {
    MeanPoolingSpec.KIND: lambda section: MeanPoolingSpec(),
    AttentionPoolingSpec.KIND: lambda section: AttentionPoolingSpec(
        section.optional("num_queries", int, AttentionPoolingSpec.num_queries)
    ),
    TransformerPoolingSpec.KIND: _parse_transformer_pooling,
}

# Each enrichment kind's parser, by its config name
_ENRICHMENT_PARSERS: dict[str, Callable[[ConfigSection], TokenWhiteningSpec]] = {
    "whitening": lambda section: TokenWhiteningSpec(
        Path(section.required("stats_path", str))
    ),
}


def _parse_group_attention(section: ConfigSection) -> GroupAttentionSpec:
    """`model.group_attention` -> GroupAttentionSpec; an absent key keeps
    the dataclass default.

    Private helper - single caller is parse_model().
    Inputs: section. Output: GroupAttentionSpec. Side effects: none.
    Exceptions: ValueError for a wrongly typed, unknown or invalid value.
    """
    defaults = GroupAttentionSpec()
    num_layers = section.optional("num_layers", int, defaults.num_layers)
    num_heads = section.optional("num_heads", int, defaults.num_heads)
    ffn_dim = section.optional("ffn_dim", int, defaults.ffn_dim)
    dropout = section.optional("dropout", float, defaults.dropout)
    norm_first = section.optional("norm_first", bool, defaults.norm_first)
    section.reject_unread_keys()
    return construct_at(
        section.where,
        lambda: GroupAttentionSpec(num_layers, num_heads, ffn_dim, dropout, norm_first),
    )


def _parse_feature_cache(section: ConfigSection) -> FeatureCacheSpec:
    """`model.feature_cache: {[cut], [storage_dtype], [vram], [ram], [disk]}`
    -> FeatureCacheSpec. With no tier key at all, the default VRAM tier;
    otherwise exactly the tiers given.

    Private helper - single caller is parse_model().
    Inputs: section. Output: FeatureCacheSpec. Side effects: none.
    Exceptions: ValueError, naming the key's location, for an unknown cut,
        storage dtype, policy or disk write mode, an unknown key, a
        missing disk path, or capacity_mib < 1.
    """
    defaults = FeatureCacheSpec()
    cut = section.optional_choice("cut", FeatureCut, defaults.cut)
    storage_dtype = section.optional_choice(
        "storage_dtype", FeatureStorageDtype, defaults.storage_dtype
    )
    # No tier named: the default; otherwise only the tiers named
    vram, ram, disk = defaults.vram, defaults.ram, defaults.disk
    if any(section.has(tier) for tier in ("vram", "ram", "disk")):
        vram = _optional_memory_tier(section, "vram")
        ram = _optional_memory_tier(section, "ram")
        disk = _optional_disk_tier(section)
    section.reject_unread_keys()
    return construct_at(
        section.where,
        lambda: FeatureCacheSpec(cut, storage_dtype, vram, ram, disk),
    )


def _optional_memory_tier(section: ConfigSection, key: str) -> MemoryTierSpec | None:
    """`{capacity_mib, [policy], [admission]}` at key -> MemoryTierSpec, or None when
    the key is absent.

    Private helper - single caller is _parse_feature_cache (vram, ram).
    Inputs: section (the feature_cache section), key. Output:
        MemoryTierSpec or None. Side effects: marks the keys read.
    Exceptions: ValueError for a missing capacity_mib, an unknown policy,
        admission or key, or capacity_mib < 1.
    """
    if not section.has(key):
        return None
    tier = section.required_section(key)
    capacity_mib = tier.required("capacity_mib", int)
    policy = tier.optional_choice("policy", SlotPolicyKind, SlotPolicyKind.LRU)
    admission = tier.optional_choice(
        "admission", AdmissionKind, AdmissionKind.FIRST_OFFER
    )
    tier.reject_unread_keys()
    return construct_at(
        tier.where, lambda: MemoryTierSpec(capacity_mib, policy, admission)
    )


def _optional_disk_tier(section: ConfigSection) -> DiskTierSpec | None:
    """`disk: {path, capacity_mib, [write]}` -> DiskTierSpec, or None when
    absent.

    Private helper - single caller is _parse_feature_cache.
    Inputs: section (the feature_cache section). Output: DiskTierSpec or
        None. Side effects: marks the keys read.
    Exceptions: ValueError for a missing path or capacity_mib, an unknown
        write mode or key, or capacity_mib < 1.
    """
    if not section.has("disk"):
        return None
    tier = section.required_section("disk")
    path = Path(tier.required("path", str))
    capacity_mib = tier.required("capacity_mib", int)
    write = tier.optional_choice("write", DiskWrite, DiskWrite.WARM_ONLY)
    tier.reject_unread_keys()
    return construct_at(tier.where, lambda: DiskTierSpec(path, capacity_mib, write))
