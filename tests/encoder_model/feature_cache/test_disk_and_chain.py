"""The durable and chained stores: SqliteFeatureStore, TieredFeatureStore
(write-through, promotion, ordering) and FrozenStageFingerprint."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.feature_store import TierStats
from src.encoder_model.feature_cache.sqlite_feature_store import SqliteFeatureStore
from src.encoder_model.feature_cache.tiered_feature_store import TieredFeatureStore
from src.encoder_model.text_encoder import TextEncoderSource
from tests.encoder_model.feature_cache.cache_fakes import CPU, fingerprint, vram_slab

WIDTH = 4


def _keys(*texts: str) -> list[CardTextKey]:
    return [CardTextKey.of_text(text) for text in texts]


def _rows(*values: float) -> torch.Tensor:
    return torch.tensor([[value] * WIDTH for value in values])


def _disk(path: Path, capacity: int = 100, **changes: object) -> SqliteFeatureStore:
    return SqliteFeatureStore(path, replace(fingerprint(WIDTH), **changes), capacity)


class TestSqliteFeatureStore:
    def test_building_it_creates_no_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cache" / "features.sqlite"
        _disk(path)
        assert not path.exists()

    def test_durable_rows_round_trip_and_persist_across_opens(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "cache" / "features.sqlite"
        a, b = _keys("a", "b")
        store = _disk(path)
        assert store.put_many([a, b], _rows(1, 2), durable=True) == 2
        store.close()

        reopened = _disk(path)
        lookup = reopened.lookup([b, *_keys("new"), a], CPU)
        assert lookup.hit_positions == [0, 2]
        assert lookup.hits is not None
        assert torch.equal(lookup.hits, _rows(2, 1))
        assert reopened.tier_stats() == (TierStats("disk", 2, 2, 100, 0, 0),)
        reopened.close()

    def test_non_durable_puts_are_ignored(self, tmp_path: Path) -> None:
        store = _disk(tmp_path / "f.sqlite")
        assert store.put_many(_keys("a"), _rows(1), durable=False) == 0
        assert store.lookup(_keys("a"), CPU).miss_positions == [0]
        store.close()

    def test_held_keys_are_skipped_and_capacity_counts_per_row_set(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "f.sqlite"
        a, b, c = _keys("a", "b", "c")
        store = _disk(path, capacity=2)
        store.put_many([a], _rows(1), durable=True)
        assert (
            store.put_many([a, b, c], _rows(9, 2, 3), durable=True) == 1
        )  # room for b only
        # Another fingerprint is another row set, with its own capacity
        other = _disk(path, capacity=2, pooling="other")
        assert other.put_many([a, b], _rows(5, 6), durable=True) == 2
        hits = store.lookup([a, b, c], CPU).hits
        assert hits is not None
        assert torch.equal(hits, _rows(1, 2))
        store.close()
        other.close()

    def test_bf16_rows_round_trip(self, tmp_path: Path) -> None:
        store = _disk(tmp_path / "f.sqlite", storage_dtype=FeatureStorageDtype.BF16)
        store.put_many(_keys("a"), _rows(1.5), durable=True)
        hits = store.lookup(_keys("a"), CPU).hits
        assert hits is not None
        assert hits.dtype == torch.bfloat16
        assert torch.equal(hits.float(), _rows(1.5))
        store.close()

    def test_lookups_past_one_query_batch_find_every_row(self, tmp_path: Path) -> None:
        store = _disk(tmp_path / "f.sqlite", capacity=2000)
        keys = _keys(*[str(i) for i in range(1200)])
        rows = torch.arange(1200, dtype=torch.float32)[:, None].repeat(1, WIDTH)
        assert store.put_many(keys, rows, durable=True) == 1200
        hits = store.lookup(keys, CPU).hits
        assert hits is not None
        assert torch.equal(hits, rows)
        store.close()

    @pytest.mark.parametrize(
        "source",
        [None, TextEncoderSource("local/path", None, 384, ())],
    )
    def test_a_fingerprint_without_a_pinned_revision_is_refused(
        self, tmp_path: Path, source: TextEncoderSource | None
    ) -> None:
        with pytest.raises(ValueError, match="known revision"):
            SqliteFeatureStore(
                tmp_path / "f.sqlite", fingerprint(WIDTH, source=source), 10
            )


class TestTieredFeatureStore:
    def test_puts_write_through_every_tier(self, tmp_path: Path) -> None:
        top, disk = vram_slab(4), _disk(tmp_path / "f.sqlite")
        chain = TieredFeatureStore([top, disk])
        assert chain.put_many(_keys("a"), _rows(1), durable=True) == 1
        assert [stats.rows_used for stats in chain.tier_stats()] == [1, 1]
        chain.close()

    def test_a_lower_tier_hit_is_promoted_into_the_tiers_above(
        self, tmp_path: Path
    ) -> None:
        top, disk = vram_slab(4), _disk(tmp_path / "f.sqlite")
        disk.put_many(_keys("a"), _rows(7), durable=True)
        chain = TieredFeatureStore([top, disk])

        first = chain.lookup(_keys("a"), CPU)
        assert first.hits is not None
        assert torch.equal(first.hits, _rows(7))
        chain.lookup(_keys("a"), CPU)
        assert chain.tier_stats() == (
            TierStats("vram", 1, 1, 4, 0, 0),
            TierStats("disk", 1, 1, 100, 0, 0),
        )
        chain.close()

    def test_hits_from_several_tiers_come_back_in_key_order(self) -> None:
        top, bottom = vram_slab(4), vram_slab(4)
        a, b, c, d = _keys("a", "b", "c", "d")
        top.put_many([c], _rows(3), durable=False)
        bottom.put_many([a, d], _rows(1, 4), durable=False)
        lookup = TieredFeatureStore([top, bottom]).lookup([a, b, c, d], CPU)
        assert lookup.hit_positions == [0, 2, 3]
        assert lookup.miss_positions == [1]
        assert lookup.hits is not None
        assert torch.equal(lookup.hits, _rows(1, 3, 4))

    def test_tiers_must_share_a_storage_dtype(self) -> None:
        with pytest.raises(ValueError, match="one dtype"):
            TieredFeatureStore(
                [vram_slab(), vram_slab(storage=FeatureStorageDtype.FP16)]
            )
        with pytest.raises(ValueError, match="at least one tier"):
            TieredFeatureStore([])


class TestFrozenStageFingerprint:
    def test_equal_fingerprints_share_a_digest(self) -> None:
        assert fingerprint(WIDTH).digest() == fingerprint(WIDTH).digest()

    @pytest.mark.parametrize(
        "change",
        [
            {"compute_precision": "fp16"},
            {"width": 8},
            {"storage_dtype": FeatureStorageDtype.FP16},
            {"pooling": "attention"},
            {"enrichment": "beef"},
            {"text_encoder": TextEncoderSource("org/model", "def456", 384, ())},
            {"text_encoder": TextEncoderSource("org/model", "abc123", 512, ())},
        ],
    )
    def test_any_changed_field_changes_the_digest(self, change: dict) -> None:
        assert (
            replace(fingerprint(WIDTH), **change).digest()
            != fingerprint(WIDTH).digest()
        )

    def test_describe_is_one_readable_line(self) -> None:
        assert (
            fingerprint(WIDTH).describe()
            == "org/model@abc123 len=384 mean fp32 -> 4 fp32"
        )


class TestSqliteEdges:
    def test_two_writers_on_one_file_count_only_their_own_inserts(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "f.sqlite"
        first, second = _disk(path), _disk(path)
        first.lookup(_keys("x"), CPU)  # both open and count 0 rows
        second.lookup(_keys("x"), CPU)
        assert first.put_many(_keys("a"), _rows(1), durable=True) == 1
        # second has not seen "a" yet; its insert is ignored, not counted
        assert second.put_many(_keys("a", "b"), _rows(9, 2), durable=True) == 1
        hits = first.lookup(_keys("a"), CPU).hits
        assert hits is not None
        assert torch.equal(hits, _rows(1))
        first.close()
        second.close()

    def test_a_full_row_set_writes_nothing(self, tmp_path: Path) -> None:
        store = _disk(tmp_path / "f.sqlite", capacity=1)
        store.put_many(_keys("a"), _rows(1), durable=True)
        assert store.put_many(_keys("b"), _rows(2), durable=True) == 0
        store.close()

    def test_a_closed_store_refuses_use(self, tmp_path: Path) -> None:
        store = _disk(tmp_path / "f.sqlite")
        store.close()
        with pytest.raises(RuntimeError, match="closed"):
            store.lookup(_keys("a"), CPU)


def test_model_kwargs_are_part_of_the_digest() -> None:
    sdpa = TextEncoderSource(
        "org/model", "abc123", 384, (("attn_implementation", "'sdpa'"),)
    )
    assert (
        replace(fingerprint(WIDTH), text_encoder=sdpa).digest()
        != fingerprint(WIDTH).digest()
    )
