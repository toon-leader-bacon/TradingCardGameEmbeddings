import math

import pytest
import torch

from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)


class TestCalculate:
    def test_computes_mean_per_datum_cross_entropy_over_ragged_logits(self) -> None:
        loss_calculator = PickPredictionCrossEntropyLoss()
        # Uniform logits (all zeros) over differently-sized packs -
        # cross-entropy against a uniform distribution of N options is
        # log(N).
        decoder_output = [torch.zeros(2), torch.zeros(3)]

        result = loss_calculator.calculate(decoder_output, [0, 1])

        expected = (math.log(2) + math.log(3)) / 2
        assert result.item() == pytest.approx(expected)

    def test_low_loss_for_confident_correct_logit(self) -> None:
        loss_calculator = PickPredictionCrossEntropyLoss()
        decoder_output = [torch.tensor([10.0, -10.0])]

        result = loss_calculator.calculate(decoder_output, [0])

        assert result.item() < 0.01

    def test_raises_on_mismatched_lengths(self) -> None:
        loss_calculator = PickPredictionCrossEntropyLoss()
        decoder_output = [torch.zeros(2)]

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [0, 1])
