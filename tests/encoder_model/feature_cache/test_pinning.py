"""Pins and reservations: pin-aware slot policies, TensorFeatureSlab pin
counts, TieredFeatureStore and SqliteFeatureStore pins, and
PooledFeatureCache / pipeline / model reserve and release."""

from pathlib import Path
from typing import AbstractSet, Sequence

import pytest
import torch

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.embedding_head import LinearEmbeddingHead
from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.pooled_feature_cache import (
    FeatureReservation,
    PooledFeatureCache,
)
from src.encoder_model.feature_cache.slot_policy import (
    AdmissionDecision,
    FillUntilFull,
    LeastRecentlyUsed,
)
from src.encoder_model.feature_cache.sqlite_feature_store import SqliteFeatureStore
from src.encoder_model.feature_cache.tensor_feature_slab import (
    SlabPlacement,
    TensorFeatureSlab,
)
from src.encoder_model.feature_cache.tiered_feature_store import TieredFeatureStore
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.encoder_model.single_card_model import SingleCardModel
from tests.encoder_model.fakes import HIDDEN, FakeTextEncoder
from tests.encoder_model.feature_cache.cache_fakes import (
    CPU,
    fingerprint,
    memory_cache,
    vram_slab,
)
from tests.encoder_model.feature_cache.test_cached_pipeline import CARDS

WIDTH = 2
FP32 = FeatureStorageDtype.FP32


def _keys(*texts: str) -> list[CardTextKey]:
    return [CardTextKey.of_text(text) for text in texts]


def _rows(count: int) -> torch.Tensor:
    return torch.arange(count * WIDTH, dtype=torch.float32).reshape(count, WIDTH)


def _lru_slab(capacity: int) -> TensorFeatureSlab:
    return TensorFeatureSlab(
        "vram",
        capacity,
        FP32,
        LeastRecentlyUsed(capacity),
        SlabPlacement.FOLLOW_FIRST_PUT,
    )


def _lru_cache(capacity: int) -> PooledFeatureCache:
    chain = TieredFeatureStore([_lru_slab(capacity)])
    return PooledFeatureCache(chain, fingerprint(WIDTH), durable_misses=False)


def _pool(texts: list[str]) -> torch.Tensor:
    return torch.stack([torch.full((WIDTH,), float(len(text))) for text in texts])


def _failing_pool(texts: list[str]) -> torch.Tensor:
    raise RuntimeError("the frozen stages failed")


def _pinned_rows(cache: PooledFeatureCache) -> int:
    return cache.stats().tiers[0].pinned_rows


class _EvictsEverything:
    """Test policy that breaks the contract: once full, evicts held keys,
    pinned or not."""

    def __init__(self, held: list[CardTextKey]) -> None:
        self._held = held

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        pass

    def admit(
        self,
        offered: Sequence[CardTextKey],
        free_rows: int,
        pinned: AbstractSet[CardTextKey],
    ) -> AdmissionDecision:
        evicted = list(self._held) if free_rows == 0 else []
        return AdmissionDecision(admitted=list(offered), evicted=evicted)


class TestPolicies:
    def test_fill_until_full_is_never_limited_by_pins(self) -> None:
        a, b = _keys("a", "b")
        decision = FillUntilFull().admit([b], free_rows=1, pinned={a})
        assert decision == AdmissionDecision([b], [], 0)

    def test_lru_evicts_the_oldest_unpinned_key(self) -> None:
        a, b, c = _keys("a", "b", "c")
        policy = LeastRecentlyUsed(2)
        policy.admit([a, b], free_rows=2, pinned=set())  # a is the oldest
        assert policy.admit([c], free_rows=0, pinned={a}) == AdmissionDecision(
            [c], [b], 0
        )

    def test_lru_refuses_what_only_pinned_rows_could_make_room_for(self) -> None:
        a, b, c, d = _keys("a", "b", "c", "d")
        policy = LeastRecentlyUsed(2)
        policy.admit([a, b], free_rows=2, pinned=set())
        decision = policy.admit([c, d], free_rows=0, pinned={a})
        assert decision == AdmissionDecision([c], [b], refused_for_pins=1)
        # Every held row pinned: nothing is admitted
        assert policy.admit([d], free_rows=0, pinned={a, c}).admitted == []

    def test_lru_ignores_pins_on_keys_it_does_not_hold(self) -> None:
        a, b, c = _keys("a", "b", "c")
        policy = LeastRecentlyUsed(1)
        policy.admit([a], free_rows=1, pinned=set())
        assert policy.admit([b], free_rows=0, pinned={c}) == AdmissionDecision(
            [b], [a], 0
        )


class TestSlabPins:
    def test_a_pinned_key_survives_eviction(self) -> None:
        slab = _lru_slab(2)
        a, b, c = _keys("a", "b", "c")
        slab.put_many([a, b], _rows(2), durable=False)
        slab.pin([a])
        slab.lookup([b], CPU)  # a is now the oldest, but pinned
        slab.put_many([c], _rows(1), durable=False)
        assert slab.lookup([a, b, c], CPU).miss_positions == [1]

    def test_a_full_pinned_slab_passes_offers_through_and_counts_them(self) -> None:
        slab = _lru_slab(2)
        a, b, c = _keys("a", "b", "c")
        slab.put_many([a, b], _rows(2), durable=False)
        slab.pin([a, b])
        assert slab.put_many([c], _rows(1), durable=False) == 0
        stats = slab.tier_stats()[0]
        assert (stats.pinned_rows, stats.refused_for_pins) == (2, 1)

    def test_pins_are_counted_per_reservation(self) -> None:
        slab = _lru_slab(1)
        a, b = _keys("a", "b")
        slab.put_many([a], _rows(1), durable=False)
        slab.pin([a])
        slab.pin([a])
        slab.unpin([a])
        assert slab.put_many([b], _rows(1), durable=False) == 0  # one pin left
        slab.unpin([a])
        assert slab.put_many([b], _rows(1), durable=False) == 1
        with pytest.raises(ValueError, match="no pin"):
            slab.unpin([a])

    def test_a_key_pinned_before_it_arrives_is_protected_once_held(self) -> None:
        slab = _lru_slab(1)
        a, b = _keys("a", "b")
        slab.pin([a])
        assert slab.tier_stats()[0].pinned_rows == 0
        slab.put_many([a], _rows(1), durable=False)
        assert slab.tier_stats()[0].pinned_rows == 1
        assert slab.put_many([b], _rows(1), durable=False) == 0

    def test_unpin_is_all_or_nothing(self) -> None:
        slab = _lru_slab(2)
        a, b = _keys("a", "b")
        slab.pin([a])
        with pytest.raises(ValueError, match="no pin"):
            slab.unpin([a, b])
        with pytest.raises(ValueError, match="same key twice"):
            slab.unpin([a, a])
        with pytest.raises(ValueError, match="same key twice"):
            slab.pin([b, b])
        slab.unpin([a])  # a's pin survived both failures

    def test_a_policy_that_evicts_a_pinned_key_is_refused(self) -> None:
        a, b = _keys("a", "b")
        slab = TensorFeatureSlab(
            "vram", 1, FP32, _EvictsEverything([a]), SlabPlacement.FOLLOW_FIRST_PUT
        )
        slab.put_many([a], _rows(1), durable=False)
        slab.pin([a])
        with pytest.raises(RuntimeError, match="pinned key"):
            slab.put_many([b], _rows(1), durable=False)


class TestChainAndDiskPins:
    def test_a_chain_pins_and_unpins_every_tier(self) -> None:
        top, bottom = vram_slab(4), vram_slab(4)
        chain = TieredFeatureStore([top, bottom])
        a = _keys("a")
        chain.put_many(a, _rows(1), durable=False)
        chain.pin(a)
        assert [stats.pinned_rows for stats in chain.tier_stats()] == [1, 1]
        chain.unpin(a)
        assert [stats.pinned_rows for stats in chain.tier_stats()] == [0, 0]

    def test_the_disk_ignores_pins(self, tmp_path: Path) -> None:
        disk = SqliteFeatureStore(tmp_path / "f.sqlite", fingerprint(WIDTH), 10)
        disk.unpin(_keys("never pinned"))
        disk.pin(_keys("a"))
        assert disk.tier_stats()[0].pinned_rows == 0


class TestCacheReservations:
    def test_reserve_warms_and_pins_without_counting_texts(self) -> None:
        cache = _lru_cache(4)
        reservation = cache.reserve(["a", "bb", "a"], _pool, CPU)
        assert len(reservation.keys) == 2
        stats = cache.stats()
        assert (stats.texts, stats.encoded, stats.tiers[0].pinned_rows) == (0, 2, 2)
        cache.release(reservation)
        assert _pinned_rows(cache) == 0

    def test_reserved_rows_are_kept_until_released(self) -> None:
        cache = _lru_cache(2)
        reservation = cache.reserve(["a", "bb"], _pool, CPU)
        cache.features(["ccc"], _pool, CPU)  # no room: passes through
        assert cache.stats().tiers[0].refused_for_pins == 1
        cache.release(reservation)
        cache.features(["ccc"], _pool, CPU)
        assert cache.stats().admitted == 3

    def test_equal_reservations_are_released_separately(self) -> None:
        cache = _lru_cache(4)
        first = cache.reserve(["a"], _pool, CPU)
        second = cache.reserve(["a"], _pool, CPU)
        assert first.keys == second.keys
        cache.release(first)
        assert _pinned_rows(cache) == 1
        cache.release(second)
        assert _pinned_rows(cache) == 0

    def test_a_reservation_is_released_once_and_only_by_its_cache(self) -> None:
        cache, other = _lru_cache(4), _lru_cache(4)
        reservation = cache.reserve(["a"], _pool, CPU)
        with pytest.raises(ValueError, match="not outstanding"):
            other.release(reservation)
        cache.release(reservation)
        with pytest.raises(ValueError, match="not outstanding"):
            cache.release(reservation)

    def test_no_texts_give_an_empty_reservation_that_releases_freely(self) -> None:
        cache = _lru_cache(4)
        reservation = cache.reserve([], _pool, CPU)
        assert reservation.keys == ()
        cache.release(reservation)
        cache.release(FeatureReservation(()))

    def test_a_failed_warm_unpins_and_re_raises(self) -> None:
        cache = _lru_cache(4)
        with pytest.raises(RuntimeError, match="frozen stages failed"):
            cache.reserve(["a"], _failing_pool, CPU)
        assert _pinned_rows(cache) == 0
        cache.features(["a"], _pool, CPU)
        assert _pinned_rows(cache) == 0  # the failed pin left nothing behind

    def test_a_precision_mismatch_pins_nothing(self) -> None:
        cache = memory_cache(WIDTH, precision="fp16")
        with pytest.raises(ValueError, match="fp16 compute"):
            cache.reserve(["a"], _pool, CPU)
        assert _pinned_rows(cache) == 0


class TestPipelineReservations:
    def _pipeline(self, cache: PooledFeatureCache | None) -> CardEmbeddingPipeline:
        head = LinearEmbeddingHead(HIDDEN, 4)
        return CardEmbeddingPipeline(
            FakeTextEncoder(), None, MeanPooling(HIDDEN), head, cache
        )

    def test_a_model_reserves_and_releases_through_its_pipeline(self) -> None:
        cache = memory_cache(HIDDEN)
        model = SingleCardModel(self._pipeline(cache))
        reservation = model.reserve(CARDS, "fp32")
        assert _pinned_rows(cache) == len(CARDS)
        model.release(reservation)
        assert _pinned_rows(cache) == 0

    def test_without_a_cache_reservations_are_empty(self) -> None:
        pipeline = self._pipeline(None)
        reservation = pipeline.reserve(CARDS, "fp32")
        assert reservation.keys == ()
        pipeline.release(reservation)
        with pytest.raises(ValueError, match="no cache"):
            pipeline.release(FeatureReservation(tuple(_keys("a"))))
