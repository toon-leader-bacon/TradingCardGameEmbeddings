"""A run's FeatureCacheSpec -> the PooledFeatureCache its model uses.

The composition root for the feature cache: picks each tier's store and
policy, chains them fastest first, and names the rows with a
FrozenStageFingerprint. scripts/run_training.py calls it from
build_card_pipeline; nothing in encoder_model/ imports training/, so the
spec-to-store mapping lives here.
"""

import hashlib
from pathlib import Path
from typing import Callable

import torch

from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.feature_store import FeatureStore
from src.encoder_model.feature_cache.frozen_stage_fingerprint import (
    FrozenStageFingerprint,
)
from src.encoder_model.feature_cache.pooled_feature_cache import PooledFeatureCache
from src.encoder_model.feature_cache.slot_policy import (
    FillUntilFull,
    FirstInFirstOut,
    LeastFrequentlyUsed,
    LeastRecentlyUsed,
    SecondOfferAdmission,
    SlotPolicy,
)
from src.encoder_model.feature_cache.sqlite_feature_store import SqliteFeatureStore
from src.encoder_model.feature_cache.tensor_feature_slab import (
    SlabPlacement,
    TensorFeatureSlab,
)
from src.encoder_model.feature_cache.tiered_feature_store import TieredFeatureStore
from src.encoder_model.precision import Precision
from src.encoder_model.text_encoder import TextEncoderSource
from src.training.model_spec import (
    AdmissionKind,
    DiskTierSpec,
    DiskWrite,
    FeatureCacheSpec,
    MemoryTierSpec,
    SlotPolicyKind,
    TokenWhiteningSpec,
)

# Bytes read at a time when hashing a whitening stats file
_HASH_CHUNK_BYTES = 2**20

# Each policy kind's constructor, given the tier's capacity_rows
_POLICY_BUILDERS: dict[SlotPolicyKind, Callable[[int], SlotPolicy]] = {
    SlotPolicyKind.LRU: LeastRecentlyUsed,
    SlotPolicyKind.FIFO: FirstInFirstOut,
    SlotPolicyKind.LFU: LeastFrequentlyUsed,
    SlotPolicyKind.FILL_UNTIL_FULL: lambda capacity_rows: FillUntilFull(),
}


def assemble_feature_cache(
    spec: FeatureCacheSpec,
    fingerprint: FrozenStageFingerprint,
) -> PooledFeatureCache:
    """The cache spec describes: its tiers (VRAM, then RAM, then disk, each
    when set) chained fastest first, its rows named by fingerprint.

    Inputs: spec (which tiers, their budgets and policies), fingerprint
        (every tier's width and storage dtype come from it alone).
    Output: PooledFeatureCache, empty in memory (the disk tier opens on
        first use).
    Side effects: none.
    Exceptions: ValueError if a tier's capacity holds no row, or spec has a
        disk tier and the fingerprint's text encoder has no source or no
        revision (SqliteFeatureStore refuses it).

    Example:
        >>> cache = assemble_feature_cache(spec.feature_cache, fingerprint)
    """
    tiers: list[FeatureStore[torch.Tensor]] = []

    # One store per tier set, fastest first
    if spec.vram is not None:
        tiers.append(
            _build_memory_tier(
                "vram", spec.vram, fingerprint, SlabPlacement.FOLLOW_FIRST_PUT
            )
        )
    if spec.ram is not None:
        tiers.append(
            _build_memory_tier("ram", spec.ram, fingerprint, SlabPlacement.PINNED_HOST)
        )
    if spec.disk is not None:
        tiers.append(_build_disk_tier(spec.disk, fingerprint))

    durable_misses = spec.disk is not None and spec.disk.write is DiskWrite.ALL
    result = PooledFeatureCache(TieredFeatureStore(tiers), fingerprint, durable_misses)
    return result


def build_fingerprint(
    source: TextEncoderSource | None,
    enrichment: TokenWhiteningSpec | None,
    pooling: str,
    compute_precision: Precision,
    width: int,
    storage_dtype: FeatureStorageDtype,
) -> FrozenStageFingerprint:
    """The fingerprint of the stages a run's cache stands in for.

    Inputs: source (the text encoder's), enrichment (the whitening spec,
        or None), pooling (the pooling kind, e.g. "mean"), compute_precision
        (the run's), width (the pooling's output_dim), storage_dtype
        (the cache spec's).
    Output: FrozenStageFingerprint.
    Side effects: reads the whitening stats file, when there is one.
    Exceptions: FileNotFoundError for a missing stats file.

    Example:
        >>> build_fingerprint(encoder.source(), None, "mean", "fp16", 768, FeatureStorageDtype.FP16)
    """
    enrichment_digest = (
        "none" if enrichment is None else _file_sha256(enrichment.stats_path)
    )
    return FrozenStageFingerprint(
        text_encoder=source,
        enrichment=enrichment_digest,
        pooling=pooling,
        compute_precision=compute_precision,
        width=width,
        storage_dtype=storage_dtype,
    )


def _build_memory_tier(
    name: str,
    spec: MemoryTierSpec,
    fingerprint: FrozenStageFingerprint,
    placement: SlabPlacement,
) -> TensorFeatureSlab:
    """A slab sized to spec's budget, with spec's policy.

    Private helper - single consumer is assemble_feature_cache (vram, ram).
    Inputs: name, spec, fingerprint (width, storage dtype), placement.
    Output: TensorFeatureSlab. Side effects: none.
    Exceptions: ValueError if the budget holds no row.
    """
    capacity_rows = fingerprint.storage_dtype.rows_in(
        spec.capacity_mib, fingerprint.width
    )
    policy = _build_policy(spec.policy, spec.admission, capacity_rows)
    return TensorFeatureSlab(
        name, capacity_rows, fingerprint.storage_dtype, policy, placement
    )


def _build_policy(
    kind: SlotPolicyKind, admission: AdmissionKind, capacity_rows: int
) -> SlotPolicy:
    """The SlotPolicy kind names, sized for capacity_rows, wrapped in
    SecondOfferAdmission (remembering capacity_rows first offers) when
    admission asks for it.

    Private helper - single consumer is _build_memory_tier.
    Inputs: kind, admission, capacity_rows. Output: SlotPolicy.
    Side effects: none. Exceptions: none.
    """
    result = _POLICY_BUILDERS[kind](capacity_rows)
    if admission is AdmissionKind.SECOND_OFFER:
        result = SecondOfferAdmission(result, remembered_keys=capacity_rows)
    return result


def _build_disk_tier(
    spec: DiskTierSpec, fingerprint: FrozenStageFingerprint
) -> SqliteFeatureStore:
    """The SQLite tier sized to spec's budget for this fingerprint's rows.

    Private helper - single consumer is assemble_feature_cache.
    Inputs: spec, fingerprint. Output: SqliteFeatureStore (not yet open).
    Side effects: none. Exceptions: ValueError if the budget holds no row,
        or (from SqliteFeatureStore) the fingerprint has no pinned source.
    """
    capacity_rows = fingerprint.storage_dtype.rows_in(
        spec.capacity_mib, fingerprint.width
    )
    return SqliteFeatureStore(spec.path, fingerprint, capacity_rows)


def _file_sha256(path: Path) -> str:
    """The sha256 hex of a file's bytes, read in chunks.

    Private helper - single consumer is build_fingerprint.
    Inputs: path. Output: str. Side effects: reads the file.
    Exceptions: FileNotFoundError.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stats_file:
        for chunk in iter(lambda: stats_file.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
