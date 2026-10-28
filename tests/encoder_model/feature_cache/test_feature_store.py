"""The memory-tier store pieces: CardTextKey, FeatureStorageDtype, the slot
policies, StoreLookup and TensorFeatureSlab (both placements; the pinned
ones on the GPU when there is one)."""

from typing import AbstractSet, Sequence

import pytest
import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.feature_store import StoreLookup, TierStats
from src.encoder_model.feature_cache.slot_policy import (
    AdmissionDecision,
    FillUntilFull,
    LeastRecentlyUsed,
    SlotPolicy,
)
from src.encoder_model.feature_cache.tensor_feature_slab import (
    SlabPlacement,
    TensorFeatureSlab,
)
from src.encoder_model.feature_cache.tiered_feature_store import TieredFeatureStore

FP32 = FeatureStorageDtype.FP32
CPU = torch.device("cpu")
_HAS_GPU = torch.cuda.is_available()


def _keys(*texts: str) -> list[CardTextKey]:
    return [CardTextKey.of_text(text) for text in texts]


def _slab(
    capacity: int,
    policy: SlotPolicy | None = None,
    placement: SlabPlacement = SlabPlacement.FOLLOW_FIRST_PUT,
    storage: FeatureStorageDtype = FP32,
) -> TensorFeatureSlab:
    return TensorFeatureSlab(
        "vram", capacity, storage, policy or FillUntilFull(), placement
    )


class _Greedy:
    """Test policy that breaks the contract: admits past capacity."""

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        pass

    def admit(
        self,
        offered: Sequence[CardTextKey],
        free_rows: int,
        pinned: AbstractSet[CardTextKey],
    ) -> AdmissionDecision:
        return AdmissionDecision(admitted=list(offered), evicted=[])


class _Reordering:
    """Test policy that admits every key, but in reverse order."""

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        pass

    def admit(
        self,
        offered: Sequence[CardTextKey],
        free_rows: int,
        pinned: AbstractSet[CardTextKey],
    ) -> AdmissionDecision:
        return AdmissionDecision(admitted=list(reversed(offered)), evicted=[])


class _Inventing:
    """Test policy that admits a key it was never offered."""

    def record_hits(self, keys: Sequence[CardTextKey]) -> None:
        pass

    def admit(
        self,
        offered: Sequence[CardTextKey],
        free_rows: int,
        pinned: AbstractSet[CardTextKey],
    ) -> AdmissionDecision:
        return AdmissionDecision(admitted=_keys("never offered"), evicted=[])


class TestCardTextKey:
    def test_equal_texts_give_equal_keys(self) -> None:
        assert CardTextKey.of_text('{"name":"Bolt"}') == CardTextKey.of_text(
            '{"name":"Bolt"}'
        )

    def test_a_reordered_text_is_another_key(self) -> None:
        # A key-shuffle mod changes the text, so it must change the key
        assert CardTextKey.of_text('{"a":1,"b":2}') != CardTextKey.of_text(
            '{"b":2,"a":1}'
        )

    def test_the_digest_is_sixteen_bytes(self) -> None:
        assert len(CardTextKey.of_text("x")) == 16
        with pytest.raises(ValueError, match="16 bytes"):
            CardTextKey(b"short")

    def test_a_key_matches_its_plain_bytes(self) -> None:
        # SQLite hands keys back as plain bytes; they must find the key
        key = CardTextKey.of_text("x")
        plain = bytes(key)
        assert type(plain) is bytes
        assert key == plain and hash(key) == hash(plain)
        assert {plain: 1}[key] == 1
        assert isinstance(CardTextKey(plain), CardTextKey)


class TestFeatureStorageDtype:
    @pytest.mark.parametrize(
        ("storage", "expected"),
        [
            (FeatureStorageDtype.FP32, torch.float32),
            (FeatureStorageDtype.FP16, torch.float16),
            (FeatureStorageDtype.BF16, torch.bfloat16),
        ],
    )
    def test_each_names_its_torch_dtype(
        self, storage: FeatureStorageDtype, expected: torch.dtype
    ) -> None:
        assert storage.torch_dtype == expected

    def test_rows_in_fits_rows_to_the_mebibytes(self) -> None:
        # 3 MiB of 768-wide fp32 rows (3 KiB each) is 1024 rows; fp16 doubles it
        assert FP32.rows_in(3, 768) == 1024
        assert FeatureStorageDtype.FP16.rows_in(3, 768) == 2048

    def test_rows_in_refuses_a_budget_below_one_row(self) -> None:
        with pytest.raises(ValueError, match="holds no"):
            FP32.rows_in(1, 2**20)
        with pytest.raises(ValueError, match="width"):
            FP32.rows_in(1, 0)


class TestPolicies:
    def test_fill_until_full_admits_the_first_free_rows_and_never_evicts(self) -> None:
        offered = _keys("a", "b", "c")
        decision = FillUntilFull().admit(offered, free_rows=2, pinned=set())
        assert decision == AdmissionDecision(admitted=offered[:2], evicted=[])
        assert (
            FillUntilFull().admit(_keys("a"), free_rows=0, pinned=set()).admitted == []
        )

    def test_lru_evicts_the_least_recently_used(self) -> None:
        a, b, c = _keys("a", "b", "c")
        policy = LeastRecentlyUsed(2)
        policy.admit([a, b], free_rows=2, pinned=set())
        policy.record_hits([a])  # b is now the oldest
        assert policy.admit([c], free_rows=0, pinned=set()) == AdmissionDecision(
            [c], [b]
        )

    def test_lru_admits_at_most_its_capacity_first_offered_first(self) -> None:
        offered = _keys("a", "b", "c")
        assert (
            LeastRecentlyUsed(2).admit(offered, free_rows=2, pinned=set()).admitted
            == offered[:2]
        )

    def test_lru_needs_a_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity_rows"):
            LeastRecentlyUsed(0)


class TestStoreLookup:
    def test_hits_must_be_none_exactly_without_hit_positions(self) -> None:
        with pytest.raises(ValueError, match="None exactly"):
            StoreLookup([0], None, [])
        with pytest.raises(ValueError, match="None exactly"):
            StoreLookup([], torch.zeros(0, 2), [0])


class TestTensorFeatureSlab:
    def test_capacity_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="capacity_rows"):
            _slab(0)

    def test_an_empty_slab_misses_everything(self) -> None:
        slab = _slab(4)
        assert slab.lookup(_keys("a", "b"), CPU) == StoreLookup([], None, [0, 1])
        assert slab.tier_stats() == (TierStats("vram", 0, 0, 4, 0, 0),)

    def test_put_rows_come_back_in_lookup_order(self) -> None:
        slab = _slab(4)
        a, b, c = _keys("a", "b", "c")
        rows = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        assert slab.put_many([a, b, c], rows, durable=False) == 3

        lookup = slab.lookup([c, CardTextKey.of_text("new"), a], CPU)
        assert lookup.hit_positions == [0, 2]
        assert lookup.miss_positions == [1]
        assert lookup.hits is not None
        assert torch.equal(lookup.hits, torch.stack([rows[2], rows[0]]))
        assert slab.tier_stats() == (TierStats("vram", 2, 3, 4, 0, 0),)

    def test_counts_evictions(self) -> None:
        slab = _slab(2, LeastRecentlyUsed(2))
        slab.put_many(_keys("a", "b"), torch.zeros(2, 2), durable=False)
        slab.put_many(_keys("c"), torch.zeros(1, 2), durable=False)
        assert slab.tier_stats()[0].evicted == 1

    def test_held_keys_are_skipped_not_refused(self) -> None:
        slab = _slab(4)
        a, b = _keys("a", "b")
        slab.put_many([a], torch.tensor([[1.0]]), durable=False)
        assert slab.put_many([a, b], torch.tensor([[9.0], [2.0]]), durable=False) == 1
        hits = slab.lookup([a, b], CPU).hits
        assert hits is not None
        assert hits.flatten().tolist() == [1.0, 2.0]  # a kept its first row

    def test_rows_are_stored_in_the_storage_dtype(self) -> None:
        slab = _slab(2, storage=FeatureStorageDtype.FP16)
        key = CardTextKey.of_text("a")
        slab.put_many([key], torch.tensor([[1.0 / 3.0, 2.0]]), durable=False)
        hits = slab.lookup([key], CPU).hits
        assert hits is not None
        assert hits.dtype == torch.float16
        assert slab.storage_dtype == torch.float16

    def test_a_full_slab_admits_only_what_fits(self) -> None:
        slab = _slab(2)
        a, b, c = _keys("a", "b", "c")
        assert slab.put_many([a, b, c], torch.ones(3, 2), durable=False) == 2
        assert slab.lookup([a, b, c], CPU).miss_positions == [2]
        assert slab.put_many([c], torch.ones(1, 2), durable=False) == 0

    def test_lru_eviction_reuses_rows(self) -> None:
        slab = _slab(2, LeastRecentlyUsed(2))
        a, b, c = _keys("a", "b", "c")
        slab.put_many([a, b], torch.tensor([[1.0], [2.0]]), durable=False)
        slab.lookup([a], CPU)  # b becomes the oldest
        assert slab.put_many([c], torch.tensor([[3.0]]), durable=False) == 1

        lookup = slab.lookup([a, b, c], CPU)
        assert lookup.miss_positions == [1]
        assert lookup.hits is not None
        assert lookup.hits.flatten().tolist() == [1.0, 3.0]

    def test_a_policy_admitting_past_capacity_is_refused(self) -> None:
        slab = _slab(1, _Greedy())
        with pytest.raises(RuntimeError, match="room for 1"):
            slab.put_many(_keys("a", "b"), torch.ones(2, 2), durable=False)

    @pytest.mark.parametrize("policy", [_Reordering(), _Inventing()])
    def test_a_policy_admitting_other_than_a_subsequence_leaves_the_slab_unchanged(
        self, policy: SlotPolicy
    ) -> None:
        slab = _slab(4, policy)
        with pytest.raises(RuntimeError, match="subsequence"):
            slab.put_many(_keys("a", "b"), torch.ones(2, 2), durable=False)
        assert slab.tier_stats()[0].rows_used == 0

    @pytest.mark.parametrize(
        ("keys", "features", "message"),
        [
            (["a", "b"], torch.ones(1, 2), "2 keys but 1"),
            (["a"], torch.ones(2), "2-D"),
            (["a", "a"], torch.ones(2, 2), "same key twice"),
            (["a"], torch.ones(1, 3), "3 wide"),
        ],
    )
    def test_a_malformed_offer_is_refused(
        self, keys: list[str], features: torch.Tensor, message: str
    ) -> None:
        slab = _slab(4)
        slab.put_many(_keys("held"), torch.ones(1, 2), durable=False)
        with pytest.raises(ValueError, match=message):
            slab.put_many(_keys(*keys), features, durable=False)

    def test_a_pinned_slab_keeps_rows_on_the_host(self) -> None:
        slab = _slab(4, placement=SlabPlacement.PINNED_HOST)
        key = CardTextKey.of_text("a")
        slab.put_many([key], torch.tensor([[1.0, 2.0]]), durable=False)
        hits = slab.lookup([key], CPU).hits
        assert hits is not None
        assert hits.device == CPU
        assert hits.tolist() == [[1.0, 2.0]]


@pytest.mark.skipif(not _HAS_GPU, reason="needs a CUDA (or ROCm) device")
class TestSlabsOnTheGpu:
    def test_a_vram_slab_follows_its_first_put_and_answers_on_any_device(self) -> None:
        cuda = torch.device("cuda")
        slab = _slab(4)
        key = CardTextKey.of_text("a")
        slab.put_many([key], torch.tensor([[1.0, 2.0]], device=cuda), durable=False)
        assert slab.lookup([key], cuda).hits.device.type == "cuda"  # type: ignore[union-attr]
        hits = slab.lookup([key], CPU).hits
        assert hits is not None
        assert hits.device == CPU

    def test_a_pinned_slab_takes_cuda_rows_and_answers_on_cuda(self) -> None:
        cuda = torch.device("cuda")
        slab = _slab(4, placement=SlabPlacement.PINNED_HOST)
        key = CardTextKey.of_text("a")
        slab.put_many([key], torch.tensor([[1.0, 2.0]], device=cuda), durable=False)
        hits = slab.lookup([key], cuda).hits
        assert hits is not None
        assert hits.device.type == "cuda"
        assert hits.tolist() == [[1.0, 2.0]]

    def test_a_side_stream_caller_reads_rows_written_on_another(self) -> None:
        # The worker writes on its own stream; the trainer reads on the default
        cuda = torch.device("cuda")
        slab = _slab(256)
        keys = _keys(*[str(i) for i in range(256)])
        rows = torch.arange(256 * 64, dtype=torch.float32, device=cuda).reshape(256, 64)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            written = rows * 2
            slab.put_many(keys, written, durable=False)
        hits = slab.lookup(keys, cuda).hits
        assert hits is not None
        assert torch.equal(hits, rows * 2)


@pytest.mark.skipif(not _HAS_GPU, reason="needs a CUDA (or ROCm) device")
class TestSlabStreamOrderingOnTheGpu:
    def test_a_side_stream_lookup_gets_rows_safe_to_read_there(self) -> None:
        cuda = torch.device("cuda")
        slab = _slab(64)
        keys = _keys(*[str(i) for i in range(64)])
        rows = torch.randn(64, 32, device=cuda)
        slab.put_many(keys, rows, durable=False)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            hits = slab.lookup(keys, cuda).hits
            assert hits is not None
            total = hits.sum()
        side.synchronize()
        assert torch.allclose(total, rows.sum())

    def test_a_gather_is_not_overwritten_by_a_later_eviction(self) -> None:
        cuda = torch.device("cuda")
        slab = _slab(1, LeastRecentlyUsed(1))
        old, new = _keys("old", "new")
        slab.put_many([old], torch.full((1, 4096), 1.0, device=cuda), durable=False)
        # The gather is issued from a side stream, the evicting put from the
        # default one: only the slab's stream ordering keeps the old row
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            hits = slab.lookup([old], cuda).hits
        slab.put_many([new], torch.full((1, 4096), 2.0, device=cuda), durable=False)
        side.synchronize()
        assert hits is not None
        assert torch.equal(hits, torch.full((1, 4096), 1.0, device=cuda))

    def test_a_pinned_ram_hit_is_promoted_into_vram_on_cuda(self) -> None:
        cuda = torch.device("cuda")
        vram = _slab(4)
        ram = _slab(4, placement=SlabPlacement.PINNED_HOST)
        key = CardTextKey.of_text("a")
        ram.put_many([key], torch.tensor([[3.0, 4.0]]), durable=False)
        chain = TieredFeatureStore([vram, ram])
        hits = chain.lookup([key], cuda).hits
        assert hits is not None
        assert hits.device.type == "cuda"
        promoted = vram.lookup([key], cuda).hits
        assert promoted is not None
        assert promoted.tolist() == [[3.0, 4.0]]
