"""PooledFeatureCache: lookups, misses encoded once, duplicates folded,
rounding through the storage dtype, durability, the precision guard and
the running totals."""

from typing import Sequence

import pytest
import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.feature_store import StoreLookup, TierStats
from src.encoder_model.feature_cache.pooled_feature_cache import (
    FeatureCacheStats,
    PooledFeatureCache,
)
from tests.encoder_model.feature_cache.cache_fakes import (
    CPU,
    fingerprint,
    memory_cache,
    vram_slab,
)

WIDTH = 3


class _CountingPool:
    """Fake frozen stages: each text maps to a fixed vector from its
    characters; every call's texts are recorded."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> torch.Tensor:
        self.calls.append(list(texts))
        return torch.stack([_vector(text) for text in texts])


class _DurabilityRecorder:
    """Fake store: never hits, records each put's durable flag."""

    storage_dtype = torch.float32

    def __init__(self) -> None:
        self.durable: list[bool] = []

    def lookup(
        self, keys: Sequence[CardTextKey], device: torch.device
    ) -> StoreLookup[torch.Tensor]:
        return StoreLookup([], None, list(range(len(keys))))

    def put_many(
        self, keys: Sequence[CardTextKey], features: torch.Tensor, durable: bool
    ) -> int:
        self.durable.append(durable)
        return 0

    def pin(self, keys: Sequence[CardTextKey]) -> None:
        pass

    def unpin(self, keys: Sequence[CardTextKey]) -> None:
        pass

    def tier_stats(self) -> tuple[TierStats, ...]:
        return ()

    def close(self) -> None:
        pass


def _vector(text: str) -> torch.Tensor:
    codes = torch.tensor([float(ord(c)) for c in text])
    return torch.stack([(codes * (i + 1)).sin().mean() / 3 for i in range(WIDTH)])


def _cache(
    capacity: int = 10, storage: FeatureStorageDtype = FeatureStorageDtype.FP32
) -> PooledFeatureCache:
    return memory_cache(WIDTH, capacity, storage)


def test_misses_are_encoded_once_then_hit() -> None:
    cache, pool = _cache(), _CountingPool()
    first = cache.features(["a", "b"], pool, CPU)
    second = cache.features(["b", "c", "a"], pool, CPU)

    assert pool.calls == [["a", "b"], ["c"]]
    assert torch.equal(first, torch.stack([_vector("a"), _vector("b")]))
    assert torch.equal(second, torch.stack([_vector(t) for t in ["b", "c", "a"]]))


def test_duplicates_within_a_call_are_encoded_once_and_expanded() -> None:
    cache, pool = _cache(), _CountingPool()
    result = cache.features(["a", "b", "a", "a"], pool, CPU)

    assert pool.calls == [["a", "b"]]
    assert torch.equal(result, torch.stack([_vector(t) for t in ["a", "b", "a", "a"]]))


def test_every_hit_miss_interleaving_keeps_order() -> None:
    cache, pool = _cache(), _CountingPool()
    cache.features(["b", "d"], pool, CPU)
    texts = ["a", "b", "c", "d", "b", "e"]
    result = cache.features(texts, pool, CPU)
    assert torch.equal(result, torch.stack([_vector(t) for t in texts]))


def test_a_full_cache_still_returns_every_feature() -> None:
    cache, pool = _cache(capacity=1), _CountingPool()
    texts = ["a", "b", "c"]
    assert torch.equal(
        cache.features(texts, pool, CPU), torch.stack([_vector(t) for t in texts])
    )
    # Only "a" was stored, so the others are encoded again
    cache.features(texts, pool, CPU)
    assert pool.calls[1] == ["b", "c"]


def test_results_are_float32_rounded_through_the_storage_dtype() -> None:
    cache, pool = _cache(storage=FeatureStorageDtype.FP16), _CountingPool()
    missed = cache.features(["a"], pool, CPU)
    hit = cache.features(["a"], pool, CPU)
    rounded = _vector("a").half().float()

    # A miss and a hit agree exactly: both went through fp16
    assert missed.dtype == torch.float32
    assert torch.equal(missed, hit)
    assert torch.equal(missed[0], rounded)


@pytest.mark.parametrize(
    "storage", [FeatureStorageDtype.FP16, FeatureStorageDtype.BF16]
)
def test_a_miss_the_full_store_rejects_is_rounded_like_a_hit(
    storage: FeatureStorageDtype,
) -> None:
    cache, pool = _cache(capacity=1, storage=storage), _CountingPool()
    cache.features(["a"], pool, CPU)
    # "a" hits, "b" misses and is rejected: both still go through the storage dtype
    result = cache.features(["a", "b"], pool, CPU)
    dtype = storage.torch_dtype
    expected = torch.stack([_vector(t).to(dtype).float() for t in ["a", "b"]])
    assert torch.equal(result, expected)
    assert cache.stats().admitted == 1


def test_results_carry_no_graph() -> None:
    cache = _cache()
    weight = torch.ones(WIDTH, requires_grad=True)
    result = cache.features(
        ["a"], lambda texts: torch.stack([weight for _ in texts]), CPU
    )
    assert not result.requires_grad


def test_misses_follow_durable_misses_unless_the_call_says_otherwise() -> None:
    store = _DurabilityRecorder()
    cache = PooledFeatureCache(store, fingerprint(WIDTH), durable_misses=False)
    cache.features(["a"], _CountingPool(), CPU)
    cache.features(["b"], _CountingPool(), CPU, durable=True)
    assert store.durable == [False, True]


def test_stats_count_texts_encodes_admissions_and_tiers() -> None:
    cache, pool = _cache(capacity=2), _CountingPool()
    assert cache.stats() == FeatureCacheStats(
        0, 0, 0, (TierStats("vram", 0, 0, 2, 0, 0),)
    )
    assert cache.stats().avoided_fraction == 0.0

    cache.features(["a", "a", "b", "c"], pool, CPU)
    cache.features(["a", "b"], pool, CPU)
    stats = cache.stats()
    assert stats == FeatureCacheStats(
        texts=6, encoded=3, admitted=2, tiers=(TierStats("vram", 2, 2, 2, 0, 0),)
    )
    assert stats.avoided_fraction == pytest.approx(0.5)


def test_no_texts_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one text"):
        _cache().features([], _CountingPool(), CPU)


def test_a_call_at_another_precision_than_the_fingerprint_is_refused() -> None:
    cache = memory_cache(WIDTH, precision="bf16")
    with pytest.raises(
        ValueError, match="built for bf16 compute, but the call runs at fp32"
    ):
        cache.features(["a"], _CountingPool(), CPU)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cache.features(["a"], _CountingPool(), CPU)


def test_a_store_of_another_dtype_than_the_fingerprint_is_refused() -> None:
    with pytest.raises(ValueError, match="fingerprint says fp16"):
        PooledFeatureCache(
            vram_slab(storage=FeatureStorageDtype.FP32),
            fingerprint(WIDTH, FeatureStorageDtype.FP16),
            durable_misses=False,
        )


def test_the_precision_guard_reads_the_innermost_autocast() -> None:
    fp32_cache = memory_cache(WIDTH, precision="fp32")
    bf16_cache = memory_cache(WIDTH, precision="bf16")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="runs at bf16"):
            fp32_cache.features(["a"], _CountingPool(), CPU)
        with torch.autocast("cpu", enabled=False):
            with pytest.raises(ValueError, match="runs at fp32"):
                bf16_cache.features(["a"], _CountingPool(), CPU)
            fp32_cache.features(["a"], _CountingPool(), CPU)
