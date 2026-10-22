import math

import pytest
import torch

from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)

_PAD = float("-inf")


class TestCalculate:
    def test_computes_mean_per_datum_cross_entropy_over_padded_logits(self) -> None:
        loss_calculator = PickPredictionCrossEntropyLoss()
        # Uniform logits over packs of 2 and 3 options, the first padded
        # with -inf: cross-entropy against N uniform options is log(N), so
        # padding must not count as an option
        decoder_output = torch.tensor([[0.0, 0.0, _PAD], [0.0, 0.0, 0.0]])

        result = loss_calculator.calculate(decoder_output, [0, 1])

        expected = (math.log(2) + math.log(3)) / 2
        assert result.item() == pytest.approx(expected)

    def test_low_loss_for_confident_correct_logit(self) -> None:
        loss_calculator = PickPredictionCrossEntropyLoss()
        decoder_output = torch.tensor([[10.0, -10.0]])

        result = loss_calculator.calculate(decoder_output, [0])

        assert result.item() < 0.01

    def test_gradients_are_finite_with_padding(self) -> None:
        logits = torch.tensor([[1.0, 2.0, _PAD], [0.5, 0.0, 1.0]], requires_grad=True)

        PickPredictionCrossEntropyLoss().calculate(logits, [1, 2]).backward()

        assert logits.grad is not None
        assert bool(torch.isfinite(logits.grad).all())
        assert logits.grad[0, 2].item() == 0.0

    def test_raises_on_mismatched_lengths(self) -> None:
        loss_calculator = PickPredictionCrossEntropyLoss()

        with pytest.raises(ValueError):
            loss_calculator.calculate(torch.zeros(1, 2), [0, 1])

    def test_raises_on_a_label_outside_the_options(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            PickPredictionCrossEntropyLoss().calculate(torch.zeros(2, 3), [0, 3])
