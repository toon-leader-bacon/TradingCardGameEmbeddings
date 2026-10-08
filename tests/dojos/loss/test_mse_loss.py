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

    def test_accepts_a_list_of_scalar_tensors(self) -> None:
        decoder_output = [torch.tensor(0.0), torch.tensor(0.0)]

        result = MseLoss().calculate(decoder_output, [1.0, 3.0])  # type: ignore[arg-type]

        assert result.item() == pytest.approx(5.0)

    def test_raises_on_a_column_shaped_output(self) -> None:
        # (N, 1) against an (N,) target would silently broadcast to (N, N)
        with pytest.raises(ValueError, match="shape"):
            MseLoss().calculate(torch.zeros(2, 1), [1.0, 2.0])

    def test_error_messages_name_the_problem(self) -> None:
        with pytest.raises(ValueError, match=r"\(1\).*\(2\)"):
            MseLoss().calculate(torch.tensor([1.0]), [1.0, 2.0])
        with pytest.raises(ValueError, match="floats"):
            MseLoss().calculate(torch.tensor([1.0]), [1])  # type: ignore[list-item]

    def test_raises_on_non_float_labels(self) -> None:
        loss_calculator = MseLoss()
        decoder_output = torch.tensor([1.0])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [1])  # type: ignore[list-item]
