import pytest
import torch

from src.dojos.losses.cross_entropy_loss import CrossEntropyLoss


class TestInit:
    def test_zero_label_smoothing_is_valid(self) -> None:
        CrossEntropyLoss(label_smoothing=0.0)  # must not raise

    @pytest.mark.parametrize("label_smoothing", [-0.1, 1.0, 1.5])
    def test_out_of_range_label_smoothing_raises(self, label_smoothing: float) -> None:
        with pytest.raises(ValueError):
            CrossEntropyLoss(label_smoothing=label_smoothing)


class TestCompute:
    def test_matches_torch_functional_cross_entropy(self) -> None:
        loss = CrossEntropyLoss(label_smoothing=0.0)
        predictions = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        labels = [0, 2]

        result = loss.compute(predictions, labels)

        expected = torch.nn.functional.cross_entropy(predictions, torch.tensor(labels))
        assert torch.isclose(result, expected)

    def test_confident_correct_predictions_yield_low_loss(self) -> None:
        loss = CrossEntropyLoss(label_smoothing=0.0)
        predictions = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        labels = [0, 1]

        result = loss.compute(predictions, labels)

        assert result.item() < 0.01

    def test_result_is_a_scalar(self) -> None:
        loss = CrossEntropyLoss(label_smoothing=0.0)
        predictions = torch.randn(4, 3)
        labels = [0, 1, 2, 0]

        result = loss.compute(predictions, labels)

        assert result.dim() == 0
