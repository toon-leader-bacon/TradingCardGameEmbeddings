import random

import pytest

from src.dojos.seventeenlands.v2.SplitSampler import SplitSamplerWellOrderedLooping

SAMPLE_METHODS = ["sample_loop", "sample_recur"]


def _sample(sampler: SplitSamplerWellOrderedLooping, method_name: str, batch_size: int):
    return getattr(sampler, method_name)(batch_size)


@pytest.mark.parametrize("method_name", SAMPLE_METHODS)
class TestSplitSamplerWellOrderedLooping:
    def test_samples_a_contiguous_prefix(self, method_name: str) -> None:
        sampler = SplitSamplerWellOrderedLooping(list(range(10)), rng_seed=0)

        result = _sample(sampler, method_name, 4)

        assert result == [0, 1, 2, 3]
        assert sampler.next_idx == 4

    def test_continues_from_next_idx(self, method_name: str) -> None:
        sampler = SplitSamplerWellOrderedLooping(list(range(10)), rng_seed=0)

        first = _sample(sampler, method_name, 3)
        second = _sample(sampler, method_name, 3)

        assert first == [0, 1, 2]
        assert second == [3, 4, 5]
        assert sampler.next_idx == 6

    def test_wraps_by_taking_the_tail_then_shuffling(self, method_name: str) -> None:
        collection = list(range(5))
        sampler = SplitSamplerWellOrderedLooping(collection.copy(), rng_seed=7)
        _sample(sampler, method_name, 3)

        expected_shuffled = list(range(5))
        random.Random(7).shuffle(expected_shuffled)

        result = _sample(sampler, method_name, 4)

        assert result[:2] == [3, 4]
        assert result[2:] == expected_shuffled[:2]
        assert sampler.next_idx == 2
        assert sampler.collection == expected_shuffled

    def test_batch_larger_than_collection_spans_epochs(self, method_name: str) -> None:
        collection = list(range(4))
        sampler = SplitSamplerWellOrderedLooping(collection.copy(), rng_seed=11)

        expected_second_epoch = list(range(4))
        random.Random(11).shuffle(expected_second_epoch)

        result = _sample(sampler, method_name, 6)

        assert result[:4] == [0, 1, 2, 3]
        assert result[4:] == expected_second_epoch[:2]
        assert sampler.next_idx == 2

    def test_empty_collection_returns_empty(self, method_name: str) -> None:
        sampler = SplitSamplerWellOrderedLooping([], rng_seed=0)

        assert _sample(sampler, method_name, 5) == []
        assert sampler.next_idx == 0

    def test_zero_batch_size_returns_empty(self, method_name: str) -> None:
        sampler = SplitSamplerWellOrderedLooping(list(range(5)), rng_seed=0)

        assert _sample(sampler, method_name, 0) == []
        assert sampler.next_idx == 0
