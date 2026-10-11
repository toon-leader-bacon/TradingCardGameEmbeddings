"""CovarianceAccumulator: streamed batches give the whole set's statistics."""

import pytest
import torch

from src.encoder_model.whitening.covariance_accumulator import CovarianceAccumulator


def test_batches_give_the_mean_and_covariance_of_all_vectors() -> None:
    vectors = torch.randn(200, 5, generator=torch.Generator().manual_seed(0)) * 3 + 7
    accumulator = CovarianceAccumulator(5)
    for batch in vectors.split(37):
        accumulator.add(batch)
    accumulator.add(torch.empty(0, 5))
    assert accumulator.count == 200
    expected = torch.cov(vectors.double().T, correction=0)
    assert torch.allclose(accumulator.mean(), vectors.double().mean(dim=0))
    assert torch.allclose(accumulator.covariance(), expected)


def test_lower_precision_input_is_accumulated_in_float64() -> None:
    accumulator = CovarianceAccumulator(3)
    accumulator.add(torch.randn(10, 3, dtype=torch.bfloat16))
    assert accumulator.mean().dtype == torch.float64


def test_a_wrongly_shaped_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="vectors must be"):
        CovarianceAccumulator(3).add(torch.randn(4, 2))


def test_statistics_need_enough_vectors() -> None:
    accumulator = CovarianceAccumulator(3)
    with pytest.raises(ValueError, match="no vectors"):
        accumulator.mean()
    accumulator.add(torch.randn(1, 3))
    with pytest.raises(ValueError, match="at least 2"):
        accumulator.covariance()


def test_dim_must_be_positive() -> None:
    with pytest.raises(ValueError):
        CovarianceAccumulator(0)
