"""A run config's `model:` section -> ModelSpec (../model_spec.py)."""

from pathlib import Path
from typing import Callable

from src.training.config_parsing.config_section import (
    ConfigSection,
    construct_at,
    parse_by_kind,
)
from src.training.model_spec import (
    AttentionPoolingSpec,
    EmbeddingHeadKind,
    GroupAttentionSpec,
    MeanPoolingSpec,
    ModelSpec,
    PoolingSpec,
    TokenWhiteningSpec,
    TransformerPoolingSpec,
)


def parse_model(section: ConfigSection) -> ModelSpec:
    """{"embedding_head", "checkpoint", "card_embedding_size",
    ["mlp_hidden_dim", "mlp_num_blocks"], ["group_attention"], ["pooling"],
    ["enrichment"]} -> ModelSpec.

    Inputs: section ("model"). Output: ModelSpec.
    Side effects: none.
    Exceptions: ValueError for an unknown embedding_head, mlp_* keys on a
        head other than residual_mlp, an unknown pooling or enrichment
        kind, or an invalid size.
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
    "mean": lambda section: MeanPoolingSpec(),
    "attention": lambda section: AttentionPoolingSpec(
        section.optional("num_queries", int, AttentionPoolingSpec.num_queries)
    ),
    "transformer": _parse_transformer_pooling,
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
