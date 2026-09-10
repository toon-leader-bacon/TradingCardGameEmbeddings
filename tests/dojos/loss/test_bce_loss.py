import math

import pytest
import torch

from src.dojos.loss.bce_loss import BceLoss


class TestCalculate:
    def test_computes_binary_cross_entropy(self) -> None:
        loss_calculator = BceLoss()
        decoder_output = torch.tensor([0.0, 0.0])

        result = loss_calculator.calculate(decoder_output, [1.0, 0.0])

        # A raw logit of 0.0 is p=0.5 either way - -log(0.5) for both terms.
        assert result.item() == pytest.approx(-math.log(0.5))

    def test_low_loss_for_confident_correct_logit(self) -> None:
        loss_calculator = BceLoss()
        decoder_output = torch.tensor([10.0])

        result = loss_calculator.calculate(decoder_output, [1.0])

        assert result.item() < 0.01

    def test_raises_on_mismatched_lengths(self) -> None:
        loss_calculator = BceLoss()
        decoder_output = torch.tensor([1.0])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [1.0, 0.0])

    def test_raises_on_non_float_labels(self) -> None:
        loss_calculator = BceLoss()
        decoder_output = torch.tensor([1.0])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [1])  # type: ignore[list-item]

    def test_raises_on_non_tensor_decoder_outputs(self) -> None:
        loss_calculator = BceLoss()

        with pytest.raises(ValueError):
            loss_calculator.calculate([1.0, 2.0], [1.0, 0.0])  # type: ignore[list-item]
