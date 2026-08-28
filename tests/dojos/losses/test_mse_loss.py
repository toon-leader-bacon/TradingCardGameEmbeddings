import torch

from src.dojos.losses.mse_loss import MSELoss


class TestCompute:
    def test_matches_torch_functional_mse_loss(self) -> None:
        loss = MSELoss()
        predictions = torch.tensor([1.5, 2.0, -1.0])
        labels = [1.0, 2.5, 0.0]

        result = loss.compute(predictions, labels)

        expected = torch.nn.functional.mse_loss(predictions, torch.tensor(labels))
        assert torch.isclose(result, expected)

    def test_exact_predictions_yield_zero_loss(self) -> None:
        loss = MSELoss()
        predictions = torch.tensor([1.0, 2.0, 3.0])
        labels = [1.0, 2.0, 3.0]

        result = loss.compute(predictions, labels)

        assert result.item() == 0.0

    def test_result_is_a_scalar(self) -> None:
        loss = MSELoss()
        predictions = torch.randn(4)
        labels = [0.1, 0.2, 0.3, 0.4]

        result = loss.compute(predictions, labels)

        assert result.dim() == 0
