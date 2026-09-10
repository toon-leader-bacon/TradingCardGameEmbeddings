import pytest
import torch

from src.dojos.loss.fixed_classification_loss import FixedClassificationLoss


class TestCalculate:
    def test_zero_loss_for_confident_correct_predictions(self) -> None:
        loss_calculator = FixedClassificationLoss(["a", "b"])
        # Large positive logit on the correct class, large negative on
        # the other - drives cross-entropy toward 0.
        decoder_output = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])

        result = loss_calculator.calculate(decoder_output, ["a", "b"])

        assert result.item() == pytest.approx(0.0, abs=1e-3)

    def test_computes_cross_entropy(self) -> None:
        loss_calculator = FixedClassificationLoss(["a", "b"])
        decoder_output = torch.tensor([[0.0, 0.0]])

        result = loss_calculator.calculate(decoder_output, ["a"])

        # Uniform logits over 2 classes -> -log(1/2)
        assert result.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())

    def test_raises_on_mismatched_lengths(self) -> None:
        loss_calculator = FixedClassificationLoss(["a", "b"])
        decoder_output = torch.tensor([[1.0, 0.0]])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, ["a", "b"])

    def test_raises_on_label_outside_label_values(self) -> None:
        loss_calculator = FixedClassificationLoss(["a", "b"])
        decoder_output = torch.tensor([[1.0, 0.0]])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, ["c"])
