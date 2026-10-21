"""feature_cache_assembly: a FeatureCacheSpec -> a cache with the right
tiers, budgets and fingerprint; and pooling_label, the pooling's part of
the fingerprint."""

from pathlib import Path

import pytest
import torch

from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.text_encoder import TextEncoderSource
from src.training.feature_cache_assembly import (
    assemble_feature_cache,
    build_fingerprint,
)
from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.slot_policy import (
    FirstInFirstOut,
    LeastFrequentlyUsed,
    LeastRecentlyUsed,
    SecondOfferAdmission,
)
from src.training.feature_cache_assembly import _build_policy
from src.training.model_spec import (
    AdmissionKind,
    AttentionPoolingSpec,
    SlotPolicyKind,
    DiskTierSpec,
    DiskWrite,
    FeatureCacheSpec,
    MeanPoolingSpec,
    MemoryTierSpec,
    TokenWhiteningSpec,
    TransformerPoolingSpec,
    pooling_label,
)

SOURCE = TextEncoderSource("org/model", "abc123", 384, ())
FP16 = FeatureStorageDtype.FP16
WIDTH = 8


def _fingerprint(spec: FeatureCacheSpec, source: TextEncoderSource | None = SOURCE):
    return build_fingerprint(source, None, "mean", "fp32", WIDTH, spec.storage_dtype)


def test_each_tier_set_becomes_a_tier_with_its_budget(tmp_path: Path) -> None:
    spec = FeatureCacheSpec(
        storage_dtype=FP16,
        vram=MemoryTierSpec(1),
        ram=MemoryTierSpec(2),
        disk=DiskTierSpec(tmp_path / "f.sqlite", 4),
    )
    cache = assemble_feature_cache(spec, _fingerprint(spec))
    rows_per_mib = 2**20 // (WIDTH * 2)
    assert [(t.name, t.capacity_rows) for t in cache.stats().tiers] == [
        ("vram", rows_per_mib),
        ("ram", 2 * rows_per_mib),
        ("disk", 4 * rows_per_mib),
    ]
    cache.close()


def test_training_misses_reach_the_disk_only_with_write_all(tmp_path: Path) -> None:
    pool = lambda texts: torch.ones(len(texts), WIDTH)  # noqa: E731
    for write, expected_disk_rows in [(DiskWrite.WARM_ONLY, 0), (DiskWrite.ALL, 1)]:
        spec = FeatureCacheSpec(
            vram=None, disk=DiskTierSpec(tmp_path / f"{write.value}.sqlite", 1, write)
        )
        cache = assemble_feature_cache(spec, _fingerprint(spec))
        cache.features(["a"], pool, torch.device("cpu"))
        assert cache.stats().tiers[0].rows_used == expected_disk_rows
        cache.close()


def test_a_disk_tier_without_a_source_is_refused(tmp_path: Path) -> None:
    spec = FeatureCacheSpec(disk=DiskTierSpec(tmp_path / "f.sqlite", 1))
    with pytest.raises(ValueError, match="known revision"):
        assemble_feature_cache(spec, _fingerprint(spec, source=None))


def test_a_memory_only_cache_needs_no_source() -> None:
    spec = FeatureCacheSpec()
    cache = assemble_feature_cache(spec, _fingerprint(spec, source=None))
    assert [t.name for t in cache.stats().tiers] == ["vram"]


def test_the_fingerprint_hashes_the_whitening_stats_file(tmp_path: Path) -> None:
    stats = tmp_path / "w.pt"
    stats.write_bytes(b"stats")
    first = build_fingerprint(
        SOURCE, TokenWhiteningSpec(stats), "mean", "fp16", 8, FP16
    )
    stats.write_bytes(b"other stats")
    second = build_fingerprint(
        SOURCE, TokenWhiteningSpec(stats), "mean", "fp16", 8, FP16
    )
    assert first.enrichment != second.enrichment != "none"
    assert first.digest() != second.digest()


def test_disk_rows_written_by_one_cache_are_found_by_the_next(tmp_path: Path) -> None:
    spec = FeatureCacheSpec(
        vram=MemoryTierSpec(1), disk=DiskTierSpec(tmp_path / "f.sqlite", 1)
    )
    pool_calls: list[list[str]] = []

    def pool(texts: list[str]) -> torch.Tensor:
        pool_calls.append(texts)
        return torch.ones(len(texts), WIDTH)

    first = assemble_feature_cache(spec, _fingerprint(spec))
    first.features(["a"], pool, torch.device("cpu"), durable=True)
    first.close()
    second = assemble_feature_cache(spec, _fingerprint(spec))
    second.features(["a"], pool, torch.device("cpu"))
    assert pool_calls == [["a"]]  # the second cache started warm from disk
    assert second.stats().tiers[1].hits == 1
    second.close()


class TestPoolingLabel:
    def test_mean_is_its_kind(self) -> None:
        assert pooling_label(MeanPoolingSpec()) == "mean"

    def test_fields_and_nested_specs_are_labelled_with_their_kind(self) -> None:
        assert pooling_label(AttentionPoolingSpec(4)) == 'attention{"num_queries":4}'
        mean_inner = pooling_label(TransformerPoolingSpec(inner=MeanPoolingSpec()))
        attention_inner = pooling_label(TransformerPoolingSpec())
        assert '"inner":"mean"' in mean_inner
        assert mean_inner != attention_inner


class TestBuildPolicy:
    @pytest.mark.parametrize(
        "kind, policy_class",
        [
            (SlotPolicyKind.LRU, LeastRecentlyUsed),
            (SlotPolicyKind.FIFO, FirstInFirstOut),
            (SlotPolicyKind.LFU, LeastFrequentlyUsed),
        ],
    )
    def test_each_kind_builds_its_policy(self, kind, policy_class) -> None:
        policy = _build_policy(kind, AdmissionKind.FIRST_OFFER, 4)
        assert type(policy) is policy_class

    @pytest.mark.parametrize("kind", list(SlotPolicyKind))
    @pytest.mark.parametrize("admission", list(AdmissionKind))
    def test_every_kind_and_admission_builds(self, kind, admission) -> None:
        policy = _build_policy(kind, admission, 4)
        key = CardTextKey.of_text("a")
        for _ in range(2):
            decision = policy.admit([key], free_rows=4, pinned=set())
        assert decision.admitted == [key]

    def test_second_offer_wraps_the_policy(self) -> None:
        policy = _build_policy(SlotPolicyKind.LRU, AdmissionKind.SECOND_OFFER, 4)
        assert isinstance(policy, SecondOfferAdmission)
        key = CardTextKey.of_text("a")
        assert policy.admit([key], free_rows=4, pinned=set()).admitted == []
        assert policy.admit([key], free_rows=4, pinned=set()).admitted == [key]
