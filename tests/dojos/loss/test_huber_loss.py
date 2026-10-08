import math

import pytest
import torch

from src.dojos.loss.huber_loss import DEFAULT_HUBER_DELTA, HuberLoss, huber_values
from src.dojos.loss.mse_loss import MseLoss


class TestHuberValues:
    def test_equals_squared_error_inside_delta(self) -> None:
        errors = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert torch.equal(huber_values(errors, delta=1.0), errors**2)

    def test_is_linear_beyond_delta(self) -> None:
        # 2*delta*|a| - delta^2 at delta=1: 2|a| - 1
        result = huber_values(torch.tensor([3.0, -10.0]), delta=1.0)
        assert result.tolist() == [5.0, 19.0]

    def test_the_two_zones_meet_at_delta(self) -> None:
        delta = 1.345
        inside = huber_values(torch.tensor([delta]), delta)
        just_outside = huber_values(torch.tensor([delta + 1e-6]), delta)
        assert just_outside.item() == pytest.approx(inside.item(), abs=1e-4)

    def test_is_twice_torchs_huber_loss(self) -> None:
        errors = torch.tensor([-47.0, -3.0, -0.2, 0.0, 0.9, 2.0, 80.0])
        expected = 2 * torch.nn.HuberLoss(reduction="none", delta=1.345)(
            errors, torch.zeros_like(errors)
        )
        assert torch.allclose(huber_values(errors, 1.345), expected)

    def test_the_gradient_is_capped_at_two_delta(self) -> None:
        delta = 1.345
        errors = torch.tensor([0.5, 3.0, 47.0, -47.0], requires_grad=True)

        huber_values(errors, delta).sum().backward()

        assert errors.grad is not None
        assert errors.grad.tolist() == pytest.approx(
            [1.0, 2 * delta, 2 * delta, -2 * delta]
        )


class TestHuberLoss:
    def test_default_delta_is_hubers_1964_value(self) -> None:
        assert HuberLoss().delta == DEFAULT_HUBER_DELTA == 1.345

    @pytest.mark.parametrize("delta", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_a_bad_delta(self, delta: float) -> None:
        with pytest.raises(ValueError, match="delta"):
            HuberLoss(delta=delta)

    def test_matches_mse_while_every_error_is_inside_delta(self) -> None:
        output = torch.tensor([0.2, -0.4, 1.0])
        labels = [0.0, 0.1, 0.5]
        assert HuberLoss().calculate(output, labels).item() == pytest.approx(
            MseLoss().calculate(output, labels).item()
        )

    def test_a_large_delta_is_mse(self) -> None:
        output = torch.tensor([5.0, -3.0, 0.5])
        labels = [0.0, 1.0, 0.0]
        assert HuberLoss(delta=1e6).calculate(output, labels).item() == pytest.approx(
            MseLoss().calculate(output, labels).item()
        )

    def test_a_47_std_miss_costs_far_less_than_mse(self) -> None:
        loss = HuberLoss().calculate(torch.tensor([47.0]), [0.0]).item()
        assert loss == pytest.approx(2 * 1.345 * 47 - 1.345**2)
        assert loss < 2209 / 10

    def test_a_47_std_miss_pulls_with_two_delta_not_94(self) -> None:
        output = torch.tensor([47.0], requires_grad=True)

        HuberLoss().calculate(output, [0.0]).backward()

        assert output.grad is not None
        assert output.grad.item() == pytest.approx(2 * 1.345)

    def test_takes_the_mean_over_the_batch(self) -> None:
        result = HuberLoss(delta=1.0).calculate(torch.tensor([0.0, 3.0]), [0.0, 0.0])
        assert result.item() == pytest.approx((0.0 + 5.0) / 2)

    def test_accepts_a_list_of_scalar_tensors(self) -> None:
        output = [torch.tensor(0.5), torch.tensor(-0.5)]
        assert HuberLoss().calculate(output, [0.0, 0.0]).item() == pytest.approx(0.25)

    def test_builds_the_target_on_the_output_device(self) -> None:
        result = HuberLoss().calculate(torch.zeros(2), [1.0, 2.0])
        assert result.device == torch.zeros(1).device
        assert result.dtype == torch.float32

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            HuberLoss().calculate(torch.tensor([1.0]), [1.0, 2.0])

    def test_raises_on_non_float_labels(self) -> None:
        with pytest.raises(ValueError, match="floats"):
            HuberLoss().calculate(torch.tensor([1.0]), [1])  # type: ignore[list-item]

    def test_raises_on_a_column_shaped_output(self) -> None:
        # (N, 1) against an (N,) target would silently broadcast to (N, N)
        with pytest.raises(ValueError, match="shape"):
            HuberLoss().calculate(torch.zeros(2, 1), [1.0, 2.0])
