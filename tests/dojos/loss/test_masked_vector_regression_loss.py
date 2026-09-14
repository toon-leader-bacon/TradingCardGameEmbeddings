import pytest
import torch

from src.dojos.loss.masked_vector_regression_loss import MaskedVectorRegressionLoss


class TestCalculate:
    def test_zero_loss_when_prediction_matches_observed_bucket(self) -> None:
        loss_calculator = MaskedVectorRegressionLoss(["0", "1", "2"])
        # sigmoid(0.0) == 0.5, matching the single observed bucket's
        # take_rate exactly.
        decoder_output = torch.tensor([[0.0, 0.0, 0.0]])

        result = loss_calculator.calculate(decoder_output, [{0: 0.5}])

        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_unobserved_buckets_are_excluded_and_averaged_per_example(self) -> None:
        # Both examples predict sigmoid(0.0) == 0.5 at every bucket.
        # Example A has one observed bucket (target 1.0, squared error
        # 0.25). Example B has three observed buckets (targets 1.0,
        # 0.5, 0.5 -> squared errors 0.25, 0, 0), so its per-example
        # mean is 0.25 / 3, not 0.25 / 1 - if a buggy implementation
        # averaged over the whole batch's masked positions instead of
        # each example's own, this would come out differently
        # ((0.25 + 0.25) / (1 + 3) == 0.125 instead of the correct
        # (0.25 + 0.25/3) / 2 == 1/6).
        loss_calculator = MaskedVectorRegressionLoss(["0", "1", "2"])
        decoder_output = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        labels = [{0: 1.0}, {0: 1.0, 1: 0.5, 2: 0.5}]

        result = loss_calculator.calculate(decoder_output, labels)

        assert result.item() == pytest.approx(1.0 / 6.0, abs=1e-4)

    def test_raises_on_mismatched_lengths(self) -> None:
        loss_calculator = MaskedVectorRegressionLoss(["0", "1"])
        decoder_output = torch.tensor([[0.0, 0.0]])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [{0: 1.0}, {1: 1.0}])

    def test_raises_on_empty_label(self) -> None:
        loss_calculator = MaskedVectorRegressionLoss(["0", "1"])
        decoder_output = torch.tensor([[0.0, 0.0]])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [{}])

    def test_raises_on_bucket_index_outside_range(self) -> None:
        loss_calculator = MaskedVectorRegressionLoss(["0", "1"])
        decoder_output = torch.tensor([[0.0, 0.0]])

        with pytest.raises(ValueError):
            loss_calculator.calculate(decoder_output, [{2: 1.0}])
