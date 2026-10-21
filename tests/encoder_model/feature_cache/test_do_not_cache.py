"""The do-not-cache stamp in the cache and the pipeline: a stamped card's
text is looked up and encoded but never stored, never reserved, and
counted as uncacheable."""

import dataclasses

import pytest
import torch

from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.embedding_head import LinearEmbeddingHead
from src.encoder_model.feature_cache.pooled_feature_cache import PooledFeatureCache
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.schema.card_factory import GenericCardFactory
from tests.encoder_model.fakes import HIDDEN, FakeTextEncoder
from tests.encoder_model.feature_cache.cache_fakes import CPU, memory_cache
from tests.encoder_model.feature_cache.test_cached_pipeline import CARDS

WIDTH = 2


def _pool(texts: list[str]) -> torch.Tensor:
    return torch.stack([torch.full((WIDTH,), float(len(text))) for text in texts])


def _cache() -> PooledFeatureCache:
    return memory_cache(WIDTH)


def _rows_used(cache: PooledFeatureCache) -> int:
    return cache.stats().tiers[0].rows_used


class TestFeatures:
    def test_an_uncacheable_text_is_encoded_but_not_stored(self) -> None:
        cache = _cache()
        result = cache.features(["a", "bb"], _pool, CPU, cacheable=[True, False])
        assert torch.equal(result, _pool(["a", "bb"]))
        stats = cache.stats()
        assert (stats.encoded, stats.admitted, stats.uncacheable) == (2, 1, 1)
        assert _rows_used(cache) == 1

    def test_the_stored_row_is_the_cacheable_texts_own(self) -> None:
        cache = _cache()
        cache.features(["a", "bb"], _pool, CPU, cacheable=[False, True])
        result = cache.features(["bb"], _pool, CPU)
        assert cache.stats().encoded == 2  # "bb" hit
        assert torch.equal(result, _pool(["bb"]))

    def test_an_uncacheable_text_is_encoded_again_next_time(self) -> None:
        cache = _cache()
        for _ in range(2):
            cache.features(["a"], _pool, CPU, cacheable=[False])
        assert (cache.stats().encoded, cache.stats().uncacheable) == (2, 2)

    def test_a_stored_text_hits_even_from_a_stamped_card(self) -> None:
        cache = _cache()
        cache.features(["a"], _pool, CPU)
        cache.features(["a"], _pool, CPU, cacheable=[False])
        assert cache.stats().encoded == 1

    def test_a_text_is_cacheable_if_any_of_its_positions_is(self) -> None:
        cache = _cache()
        cache.features(["a", "a"], _pool, CPU, cacheable=[False, True])
        assert (cache.stats().admitted, cache.stats().uncacheable) == (1, 0)

    def test_nothing_is_offered_when_every_miss_is_uncacheable(self) -> None:
        cache = _cache()
        cache.features(["a", "bb"], _pool, CPU, cacheable=[False, False])
        assert _rows_used(cache) == 0

    def test_flags_must_match_the_texts(self) -> None:
        with pytest.raises(ValueError, match="cacheable flags"):
            _cache().features(["a", "bb"], _pool, CPU, cacheable=[True])


class TestReserve:
    def test_uncacheable_texts_are_neither_pinned_nor_encoded(self) -> None:
        cache = _cache()
        reservation = cache.reserve(["a", "bb"], _pool, CPU, cacheable=[True, False])
        assert len(reservation.keys) == 1
        assert (cache.stats().encoded, cache.stats().tiers[0].pinned_rows) == (1, 1)
        cache.release(reservation)

    def test_an_all_uncacheable_batch_gives_an_empty_reservation(self) -> None:
        cache = _cache()
        reservation = cache.reserve(["a"], _pool, CPU, cacheable=[False])
        assert reservation.keys == ()
        assert cache.stats().encoded == 0
        cache.release(reservation)


class TestPipeline:
    def _pipeline(self, cache: PooledFeatureCache) -> CardEmbeddingPipeline:
        head = LinearEmbeddingHead(HIDDEN, 4)
        return CardEmbeddingPipeline(
            FakeTextEncoder(), None, MeanPooling(HIDDEN), head, cache
        )

    def test_embed_never_stores_a_stamped_card(self) -> None:
        cache = memory_cache(HIDDEN)
        stamped = GenericCardFactory.stamp_do_not_cache(CARDS[0])
        self._pipeline(cache).embed([stamped, CARDS[1]])
        assert (cache.stats().admitted, cache.stats().uncacheable) == (1, 1)

    def test_reserve_skips_a_stamped_card(self) -> None:
        cache = memory_cache(HIDDEN)
        pipeline = self._pipeline(cache)
        stamped = dataclasses.replace(CARDS[0], cacheable=False)
        reservation = pipeline.reserve([stamped, CARDS[1]], "fp32")
        assert len(reservation.keys) == 1
        pipeline.release(reservation)
