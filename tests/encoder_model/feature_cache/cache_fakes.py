"""Shared builders for feature cache tests: fingerprints and small caches."""

import torch

from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.frozen_stage_fingerprint import (
    FrozenStageFingerprint,
)
from src.encoder_model.feature_cache.pooled_feature_cache import PooledFeatureCache
from src.encoder_model.feature_cache.slot_policy import FillUntilFull
from src.encoder_model.feature_cache.tensor_feature_slab import (
    SlabPlacement,
    TensorFeatureSlab,
)
from src.encoder_model.feature_cache.tiered_feature_store import TieredFeatureStore
from src.encoder_model.precision import Precision
from src.encoder_model.text_encoder import TextEncoderSource

CPU = torch.device("cpu")
PINNED_SOURCE = TextEncoderSource("org/model", "abc123", 384, ())


def fingerprint(
    width: int,
    storage: FeatureStorageDtype = FeatureStorageDtype.FP32,
    precision: Precision = "fp32",
    source: TextEncoderSource | None = PINNED_SOURCE,
) -> FrozenStageFingerprint:
    """A fingerprint for width-wide rows from a pinned fake source."""
    return FrozenStageFingerprint(source, "none", "mean", precision, width, storage)


def vram_slab(
    capacity: int = 10, storage: FeatureStorageDtype = FeatureStorageDtype.FP32
) -> TensorFeatureSlab:
    """A fill-until-full slab that follows its first put (CPU in tests)."""
    return TensorFeatureSlab(
        "vram", capacity, storage, FillUntilFull(), SlabPlacement.FOLLOW_FIRST_PUT
    )


def memory_cache(
    width: int,
    capacity: int = 10,
    storage: FeatureStorageDtype = FeatureStorageDtype.FP32,
    precision: Precision = "fp32",
) -> PooledFeatureCache:
    """A cache over one fill-until-full slab, chained as the assembly does."""
    chain = TieredFeatureStore([vram_slab(capacity, storage)])
    return PooledFeatureCache(chain, fingerprint(width, storage, precision), False)
