"""Token-level feature rows: RaggedTokenStates, the row kinds, policies
counting rows, and every store holding token rows."""

from pathlib import Path

import pytest
import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.cut_point import PooledCutPoint, TokenCutPoint
from src.encoder_model.feature_cache.feature_row_kind import (
    POOLED_ROWS,
    TOKEN_ROWS,
    require_offer_shape,
)
from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.frozen_feature_cache import FrozenFeatureCache
from src.encoder_model.feature_cache.frozen_stage_fingerprint import (
    FrozenStageFingerprint,
)
from src.encoder_model.feature_cache.ragged_token_states import (
    RaggedTokenStates,
    token_row_ranges,
)
from src.encoder_model.feature_cache.slot_policy import (
    FillUntilFull,
    FirstInFirstOut,
    LeastFrequentlyUsed,
    LeastRecentlyUsed,
    SecondOfferAdmission,
)
from src.encoder_model.feature_cache.sqlite_feature_store import SqliteFeatureStore
from src.encoder_model.feature_cache.tensor_feature_slab import (
    SlabPlacement,
    TensorFeatureSlab,
)
from src.encoder_model.feature_cache.tiered_feature_store import TieredFeatureStore
from src.encoder_model.text_encoder import TokenEncoding
from tests.encoder_model.feature_cache.cache_fakes import PINNED_SOURCE

CPU = torch.device("cpu")
WIDTH = 4
FP32 = FeatureStorageDtype.FP32


def _key(name: str) -> CardTextKey:
    return CardTextKey.of_text(name)


def _states(*lengths: int, offset: float = 0.0) -> RaggedTokenStates:
    """Texts of the given token counts; row r of the packed values is
    r + offset in every column, so any misplaced row shows."""
    total = sum(lengths)
    values = (torch.arange(total, dtype=torch.float32) + offset).unsqueeze(1)
    return RaggedTokenStates(
        values.repeat(1, WIDTH), torch.tensor(lengths, dtype=torch.int64)
    )


def _token_fingerprint(layer: int | None = 3) -> FrozenStageFingerprint:
    return FrozenStageFingerprint(
        PINNED_SOURCE, "none", "none", "fp32", WIDTH, FP32, TokenCutPoint(layer)
    )


def _token_slab(capacity_rows: int, policy: object = None) -> TensorFeatureSlab:
    return TensorFeatureSlab(
        "vram",
        capacity_rows,
        FP32,
        policy or LeastRecentlyUsed(capacity_rows),  # type: ignore[arg-type]
        SlabPlacement.FOLLOW_FIRST_PUT,
        TOKEN_ROWS,
    )


class TestRaggedTokenStates:
    def test_an_encoding_packs_and_pads_back_unchanged(self) -> None:
        hidden = torch.randn(3, 5, WIDTH)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
        states = RaggedTokenStates.from_encoding(TokenEncoding(hidden, mask))
        assert states.lengths.tolist() == [3, 5, 1]
        padded = states.to_encoding()
        assert torch.equal(padded.attention_mask, mask)
        assert torch.equal(
            padded.hidden_states * mask.unsqueeze(-1), hidden * mask.unsqueeze(-1)
        )

    def test_select_reorders_and_repeats_texts(self) -> None:
        states = _states(2, 3)
        picked = states.select([1, 0, 1])
        assert picked.lengths.tolist() == [3, 2, 3]
        assert picked.values[:, 0].tolist() == [2, 3, 4, 0, 1, 2, 3, 4]

    def test_concatenate_joins_texts_in_order(self) -> None:
        joined = RaggedTokenStates.concatenate([_states(1), _states(2, offset=10)])
        assert joined.lengths.tolist() == [1, 2]
        assert joined.values[:, 0].tolist() == [0, 10, 11]

    def test_row_ranges_cover_each_text_in_order(self) -> None:
        lengths = torch.tensor([2, 3, 1])
        assert token_row_ranges(lengths, [2, 0]).tolist() == [5, 0, 1]

    @pytest.mark.parametrize(
        "lengths", [torch.tensor([2, 2]), torch.tensor([3, 0]), torch.tensor([3.0])]
    )
    def test_lengths_must_be_positive_int64_and_cover_values(
        self, lengths: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError):
            RaggedTokenStates(torch.zeros(3, WIDTH), lengths)

    def test_a_text_without_a_real_token_cannot_be_packed(self) -> None:
        encoding = TokenEncoding(torch.zeros(1, 2, WIDTH), torch.zeros(1, 2))
        with pytest.raises(ValueError):
            RaggedTokenStates.from_encoding(encoding)


class TestRowKinds:
    def test_token_rows_count_one_row_per_token(self) -> None:
        states = _states(2, 3)
        assert TOKEN_ROWS.row_counts(states) == [2, 3]
        rebuilt = TOKEN_ROWS.from_flat_rows(TOKEN_ROWS.flat_rows(states), [2, 3])
        assert torch.equal(rebuilt.values, states.values)

    def test_pooled_rows_count_one_row_per_text(self) -> None:
        assert POOLED_ROWS.row_counts(torch.zeros(3, WIDTH)) == [1, 1, 1]
        with pytest.raises(ValueError):
            POOLED_ROWS.from_flat_rows(torch.zeros(3, WIDTH), [1, 2])

    def test_the_offer_check_returns_row_counts_and_refuses_bad_offers(self) -> None:
        keys = [_key("a"), _key("b")]
        assert require_offer_shape(TOKEN_ROWS, keys, _states(2, 3), WIDTH) == [2, 3]
        with pytest.raises(ValueError):
            require_offer_shape(TOKEN_ROWS, keys[:1], _states(2, 3), WIDTH)
        with pytest.raises(ValueError):
            require_offer_shape(TOKEN_ROWS, keys, _states(2, 3), WIDTH + 1)

    def test_each_cut_point_names_its_row_kind(self) -> None:
        assert PooledCutPoint().row_kind is POOLED_ROWS
        assert TokenCutPoint(4).row_kind is TOKEN_ROWS
        assert TokenCutPoint(4).describe() == "tokens@4"


class TestPoliciesCountRows:
    def test_fill_until_full_admits_the_prefix_whose_rows_fit(self) -> None:
        keys = [_key("a"), _key("b"), _key("c")]
        decision = FillUntilFull().admit(keys, 5, set(), row_counts=[2, 3, 1])
        assert decision.admitted == keys[:2]

    @pytest.mark.parametrize("policy_type", [FirstInFirstOut, LeastRecentlyUsed])
    def test_one_long_text_evicts_several_short_ones(self, policy_type: type) -> None:
        policy = policy_type(6)
        small = [_key(name) for name in "abc"]
        policy.admit(small, 6, set(), row_counts=[2, 2, 2])
        decision = policy.admit([_key("long")], 0, set(), row_counts=[4])
        assert decision.admitted == [_key("long")]
        assert decision.evicted == small[:2]

    def test_least_frequently_used_frees_rows_from_the_least_hit(self) -> None:
        policy = LeastFrequentlyUsed(4)
        a, b = _key("a"), _key("b")
        policy.admit([a, b], 4, set(), row_counts=[2, 2])
        policy.record_hits([a])
        decision = policy.admit([_key("c")], 0, set(), row_counts=[2])
        assert decision.evicted == [b]

    def test_pinned_rows_are_never_freed(self) -> None:
        policy = LeastRecentlyUsed(4)
        a, b = _key("a"), _key("b")
        policy.admit([a, b], 4, set(), row_counts=[2, 2])
        decision = policy.admit([_key("c")], 0, {a, b}, row_counts=[2])
        assert decision.admitted == []
        assert decision.refused_for_pins == 1

    def test_a_text_longer_than_the_tier_is_never_admitted(self) -> None:
        decision = LeastRecentlyUsed(3).admit([_key("a")], 3, set(), row_counts=[4])
        assert decision.admitted == []

    def test_second_offer_passes_the_repeats_row_counts_on(self) -> None:
        policy = SecondOfferAdmission(FillUntilFull(), remembered_keys=4)
        a = _key("a")
        policy.admit([a], 3, set(), row_counts=[4])
        assert policy.admit([a], 3, set(), row_counts=[4]).admitted == []


class TestTokenSlab:
    def test_token_rows_come_back_as_they_went_in(self) -> None:
        slab = _token_slab(10)
        keys = [_key("a"), _key("b"), _key("c")]
        states = _states(2, 3, 1)
        assert slab.put_many(keys, states, durable=False) == 3
        found = slab.lookup([keys[2], _key("x"), keys[0]], CPU)
        assert found.hit_positions == [0, 2]
        assert found.miss_positions == [1]
        assert found.hits is not None
        assert found.hits.lengths.tolist() == [1, 2]
        assert found.hits.values[:, 0].tolist() == [5, 0, 1]

    def test_evicted_rows_are_reused_without_fragmenting(self) -> None:
        slab = _token_slab(5, FirstInFirstOut(5))
        a, b, c = _key("a"), _key("b"), _key("c")
        slab.put_many([a, b], _states(2, 3), durable=False)
        # The slab is full; c's 4 rows need a's 2 and some of b's 3, so
        # FIFO evicts both, and c's rows are scattered over theirs
        slab.put_many([c], _states(4, offset=100), durable=False)
        found = slab.lookup([a, b, c], CPU)
        assert found.hit_positions == [2]
        assert found.hits is not None
        assert found.hits.values[:, 0].tolist() == [100, 101, 102, 103]
        stats = slab.tier_stats()[0]
        assert (stats.rows_used, stats.evicted) == (4, 2)

    def test_pinned_rows_count_their_tokens(self) -> None:
        slab = _token_slab(10)
        a = _key("a")
        slab.pin([a])
        slab.put_many([a], _states(3), durable=False)
        assert slab.tier_stats()[0].pinned_rows == 3


class TestTokenDiskAndChain:
    def test_the_disk_tier_stores_one_blob_per_text(self, tmp_path: Path) -> None:
        store = SqliteFeatureStore(tmp_path / "f.sqlite", _token_fingerprint(), 100)
        keys = [_key("a"), _key("b")]
        assert store.put_many(keys, _states(2, 3), durable=True) == 2
        found = store.lookup([keys[1], keys[0]], CPU)
        assert found.hits is not None
        assert found.hits.lengths.tolist() == [3, 2]
        assert found.hits.values[:, 0].tolist() == [2, 3, 4, 0, 1]
        assert store.tier_stats()[0].rows_used == 5
        store.close()

    def test_the_disk_tier_counts_its_capacity_in_tokens(self, tmp_path: Path) -> None:
        store = SqliteFeatureStore(tmp_path / "f.sqlite", _token_fingerprint(), 4)
        keys = [_key("a"), _key("b")]
        assert store.put_many(keys, _states(2, 3), durable=True) == 1
        store.close()
        reopened = SqliteFeatureStore(tmp_path / "f.sqlite", _token_fingerprint(), 4)
        reopened.lookup([keys[0]], CPU)
        assert reopened.tier_stats()[0].rows_used == 2
        reopened.close()

    def test_a_chain_merges_token_hits_from_several_tiers(self, tmp_path: Path) -> None:
        top = _token_slab(10)
        disk = SqliteFeatureStore(tmp_path / "f.sqlite", _token_fingerprint(), 100)
        chain = TieredFeatureStore([top, disk])
        a, b = _key("a"), _key("b")
        disk.put_many([a], _states(2), durable=True)
        top.put_many([b], _states(3, offset=50), durable=False)
        found = chain.lookup([a, b], CPU)
        assert found.hit_positions == [0, 1]
        assert found.hits is not None
        assert found.hits.values[:, 0].tolist() == [0, 1, 50, 51, 52]
        chain.close()

    def test_a_chain_refuses_tiers_of_different_row_kinds(self) -> None:
        pooled = TensorFeatureSlab(
            "ram", 4, FP32, FillUntilFull(), SlabPlacement.PINNED_HOST, POOLED_ROWS
        )
        with pytest.raises(ValueError, match="row kind"):
            TieredFeatureStore([_token_slab(4), pooled])

    def test_a_cache_refuses_a_store_that_cannot_hold_its_cut(self) -> None:
        with pytest.raises(ValueError, match="row kind"):
            FrozenFeatureCache(
                TieredFeatureStore([_token_slab(4)]),
                FrozenStageFingerprint(
                    PINNED_SOURCE, "none", "mean", "fp32", WIDTH, FP32
                ),
                durable_misses=False,
            )

    def test_a_cache_returns_float32_token_states_for_hits_and_misses(self) -> None:
        cache = FrozenFeatureCache(
            TieredFeatureStore([_token_slab(20)]), _token_fingerprint(), False
        )
        encoded: list[list[str]] = []

        def encode(texts: list[str]) -> RaggedTokenStates:
            encoded.append(texts)
            lengths = [len(text) for text in texts]
            return _states(*lengths, offset=float(len(encoded) * 100))

        first = cache.features(["ab", "abc"], encode, CPU)
        second = cache.features(["abc", "x", "ab", "abc"], encode, CPU)
        assert encoded == [["ab", "abc"], ["x"]]
        assert second.lengths.tolist() == [3, 1, 2, 3]
        assert second.values.dtype == torch.float32
        assert torch.equal(second.select([2]).values, first.select([0]).values)


class TestTokenFingerprint:
    def test_a_pooled_digest_is_unchanged_by_the_cut_field(self) -> None:
        pooled = FrozenStageFingerprint(PINNED_SOURCE, "none", "mean", "fp32", 8, FP32)
        explicit = FrozenStageFingerprint(
            PINNED_SOURCE, "none", "mean", "fp32", 8, FP32, PooledCutPoint()
        )
        assert pooled.digest() == explicit.digest()

    def test_each_cut_layer_names_its_own_row_set(self) -> None:
        assert _token_fingerprint(3).digest() != _token_fingerprint(4).digest()
        assert _token_fingerprint(None).digest() != _token_fingerprint(3).digest()
        assert "tokens@3" in _token_fingerprint(3).describe()

    def test_a_token_cut_records_no_enrichment_or_pooling(self) -> None:
        with pytest.raises(ValueError, match="token cut"):
            FrozenStageFingerprint(
                PINNED_SOURCE, "none", "mean", "fp32", WIDTH, FP32, TokenCutPoint(2)
            )
