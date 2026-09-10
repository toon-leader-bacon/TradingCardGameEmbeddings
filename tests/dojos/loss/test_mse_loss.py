import pytest
import torch

from src.dojos.loss.mse_loss import MseLoss


class TestCalculate:
    def test_zero_loss_for_perfect_predictions(self) -> None:
        loss_calculator = MseLoss()
        decoder_output = torch.tensor([1.0, 2.0, 3.0])

        result = loss_calculator.calculate(decoder_output, [1.0, 2.0, 3.0])

        assert result.item() == pytest.approx(0.0)

    def test_computes_mean_squared_error(self) -> None:
        loss_calculator = MseLoss()
        decoder_output = torch.tensor([0.0, 0.0])

        result = loss_calculator.calculate(decoder_output, [1.0, 3.0])

        assert result.item() == pytest.approx((1.0**2 + 3.0**2) / 2)

    def test_raises_on_mismatched_lengths(self) -> None:
        loss_calculator = MseLoss()
        decoder_output = torch.tensor([1.0])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [1.0, 2.0])

    def test_raises_on_non_float_labels(self) -> None:
        loss_calculator = MseLoss()
        decoder_output = torch.tensor([1.0])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [1])  # type: ignore[list-item]
