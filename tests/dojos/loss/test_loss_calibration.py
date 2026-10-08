import math
from typing import Any

import pytest
import torch

from src.dojos.loss.fixed_classification_loss import FixedClassificationLoss
from src.dojos.loss.label_stats import LabelStats
from src.dojos.loss.huber_loss import HuberLoss, huber_values
from src.dojos.loss.loss_calibration import (
    CalibratedLoss,
    HuberRegressionCalibration,
    StandardizedRegressionCalibration,
    best_constant_huber_loss,
)
from src.dojos.loss.mse_loss import MseLoss
from src.dojos.loss.standardized_label_loss import StandardizedLabelLoss

_CARD: Any = object()  # inputs are never read by these calibrations


class TestStandardizedLabelLoss:
    def test_a_prediction_at_the_standardized_label_scores_zero(self) -> None:
        loss = StandardizedLabelLoss(MseLoss(), LabelStats(2.0, 1.0))
        assert loss.calculate(torch.tensor([0.0]), [2.0]).item() == 0.0

    def test_scores_in_standard_deviation_units(self) -> None:
        loss = StandardizedLabelLoss(MseLoss(), LabelStats(mean=100.0, std=10.0))
        # Label 120 is z = 2; predicting 0 costs 2^2, not 120^2
        assert loss.calculate(torch.tensor([0.0]), [120.0]).item() == 4.0

    def test_inner_validation_still_applies(self) -> None:
        loss = StandardizedLabelLoss(MseLoss(), LabelStats(0.0, 1.0))
        with pytest.raises(ValueError):
            loss.calculate(torch.tensor([0.0, 1.0]), [1.0])


class TestCalibratedLoss:
    @pytest.mark.parametrize("baseline", [0.0, -0.0, -1.0, math.nan, math.inf])
    def test_rejects_an_unusable_baseline(self, baseline: float) -> None:
        with pytest.raises(ValueError, match="baseline_loss"):
            CalibratedLoss(MseLoss(), baseline)

    def test_label_stats_are_read_off_a_standardizing_loss(self) -> None:
        stats = LabelStats(2.0, 1.0)
        calibrated = CalibratedLoss(StandardizedLabelLoss(MseLoss(), stats), 1.0)
        assert calibrated.label_stats == stats

    def test_any_other_loss_has_no_label_stats(self) -> None:
        assert CalibratedLoss(FixedClassificationLoss(["a"]), 0.7).label_stats is None


class TestStandardizedRegressionCalibration:
    def test_wraps_the_loss_with_train_stats_and_a_unit_baseline(self) -> None:
        calibrated = StandardizedRegressionCalibration().calibrate(
            MseLoss(), [(_CARD, 1.0), (_CARD, 3.0)]
        )
        assert isinstance(calibrated.loss, StandardizedLabelLoss)
        assert calibrated.label_stats == LabelStats(mean=2.0, std=1.0)
        assert calibrated.baseline_loss == 1.0

    def test_the_mean_predictor_scores_the_baseline_on_train(self) -> None:
        labels = [450.0, 300.0, 610.0, 512.0, 97.0]
        calibrated = StandardizedRegressionCalibration().calibrate(
            MseLoss(), [(_CARD, label) for label in labels]
        )
        mean_prediction = torch.zeros(len(labels))  # 0 in standardized units
        loss = calibrated.loss.calculate(mean_prediction, labels).item()
        assert math.isclose(loss, calibrated.baseline_loss, rel_tol=1e-5)

    def test_a_constant_label_is_a_construction_error(self) -> None:
        with pytest.raises(ValueError, match="zero standard deviation"):
            StandardizedRegressionCalibration().calibrate(
                MseLoss(), [(_CARD, 4.0), (_CARD, 4.0)]
            )

    def test_an_empty_sample_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            StandardizedRegressionCalibration().calibrate(MseLoss(), [])

    def test_a_non_float_label_is_refused(self) -> None:
        with pytest.raises(TypeError, match="float labels"):
            StandardizedRegressionCalibration().calibrate(
                MseLoss(), [(_CARD, "monster"), (_CARD, "beast")]
            )


def _grid_minimum(labels: list[float], delta: float) -> float:
    """Brute force: the lowest mean Huber loss over a fine grid of constants."""
    values = torch.tensor(labels, dtype=torch.float64)
    grid = torch.linspace(min(labels) - 1, max(labels) + 1, 4001, dtype=torch.float64)
    losses = [huber_values(values - c, delta).mean().item() for c in grid]
    return min(losses)


class TestBestConstantHuberLoss:
    def test_symmetric_labels_have_their_center_as_the_best_constant(self) -> None:
        assert best_constant_huber_loss([-1.0, 1.0], delta=1.345) == pytest.approx(1.0)

    def test_matches_a_brute_force_search_over_constants(self) -> None:
        labels = [-0.4, -0.3, 0.1, 0.2, 0.3, 0.5, 0.6, 9.0, -0.1, 0.0]
        found = best_constant_huber_loss(labels, delta=1.345)
        assert found == pytest.approx(_grid_minimum(labels, 1.345), abs=1e-4)
        assert found <= _grid_minimum(labels, 1.345) + 1e-9

    def test_is_never_worse_than_the_mean_predictor(self) -> None:
        labels = [0.0, 0.1, -0.2, 0.3, 40.0, -0.1]
        mean = sum(labels) / len(labels)
        values = torch.tensor(labels, dtype=torch.float64)
        at_mean = huber_values(values - mean, 1.345).mean().item()
        assert best_constant_huber_loss(labels, 1.345) <= at_mean + 1e-12

    def test_a_heavy_tail_gives_a_baseline_far_below_one(self) -> None:
        # z-scored: 999 labels near zero and a single 30-std outlier
        raw = [0.0] * 999 + [1000.0]
        stats = LabelStats.from_labels(raw)
        baseline = best_constant_huber_loss(stats.standardize(raw), 1.345)
        assert baseline < 0.2

    def test_gaussian_like_labels_stay_close_to_the_quadratic_value(self) -> None:
        torch.manual_seed(0)
        labels = torch.randn(5000).tolist()
        assert best_constant_huber_loss(labels, 1.345) == pytest.approx(1.0, abs=0.1)

    def test_a_huge_delta_is_the_mean_predictors_mse(self) -> None:
        labels = [1.0, 2.0, 6.0]
        mean = 3.0
        expected = sum((x - mean) ** 2 for x in labels) / 3
        assert best_constant_huber_loss(labels, 1e9) == pytest.approx(expected)

    def test_the_baseline_is_the_loss_at_the_best_constant(self) -> None:
        labels = [-0.4, 0.2, 0.3, 9.0, -0.1]
        baseline = best_constant_huber_loss(labels, 1.345)
        # Predicting the best constant everywhere scores the baseline
        center = _best_center(labels, 1.345)
        output = torch.full((len(labels),), center, dtype=torch.float64)
        loss = HuberLoss(1.345).calculate(output, labels)
        assert loss.item() == pytest.approx(baseline, abs=1e-4)

    def test_discrete_tied_labels_with_a_tail_at_a_small_delta(self) -> None:
        # Count-like labels: many ties at 0 and 1, a thin tail
        labels = [-0.3] * 60 + [0.4] * 25 + [1.1] * 10 + [8.0, 12.0, 20.0]
        found = best_constant_huber_loss(labels, delta=0.1)
        assert found == pytest.approx(_grid_minimum(labels, 0.1), abs=1e-4)

    @pytest.mark.parametrize("delta", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_a_bad_delta(self, delta: float) -> None:
        with pytest.raises(ValueError, match="delta"):
            best_constant_huber_loss([1.0, 2.0], delta)

    def test_rejects_empty_and_non_finite_labels(self) -> None:
        with pytest.raises(ValueError, match="no labels"):
            best_constant_huber_loss([], 1.0)
        with pytest.raises(ValueError, match="finite"):
            best_constant_huber_loss([1.0, math.nan], 1.0)


def _best_center(labels: list[float], delta: float) -> float:
    values = torch.tensor(labels, dtype=torch.float64)
    grid = torch.linspace(min(labels), max(labels), 20001, dtype=torch.float64)
    losses = torch.stack([huber_values(values - c, delta).mean() for c in grid])
    return float(grid[int(losses.argmin())])


class TestHuberRegressionCalibration:
    def test_wraps_the_loss_with_train_stats_and_a_computed_baseline(self) -> None:
        calibrated = HuberRegressionCalibration().calibrate(
            HuberLoss(), [(_CARD, 1.0), (_CARD, 3.0)]
        )
        assert isinstance(calibrated.loss, StandardizedLabelLoss)
        assert calibrated.label_stats == LabelStats(mean=2.0, std=1.0)
        assert calibrated.baseline_loss == pytest.approx(1.0)

    def test_the_baseline_uses_the_losss_own_delta(self) -> None:
        sample = [(_CARD, float(x)) for x in [0, 0, 0, 0, 0, 0, 0, 0, 0, 50]]
        narrow = HuberRegressionCalibration().calibrate(HuberLoss(0.1), sample)
        wide = HuberRegressionCalibration().calibrate(HuberLoss(100.0), sample)
        assert narrow.baseline_loss < wide.baseline_loss

    def test_a_heavy_tail_baseline_is_below_the_mse_baseline(self) -> None:
        sample = [(_CARD, 0.0)] * 499 + [(_CARD, 500.0)]
        calibrated = HuberRegressionCalibration().calibrate(HuberLoss(), sample)
        assert calibrated.baseline_loss < 0.2

    def test_rejects_a_loss_that_is_not_a_huber_loss(self) -> None:
        with pytest.raises(TypeError, match="HuberLoss"):
            HuberRegressionCalibration().calibrate(
                MseLoss(), [(_CARD, 1.0), (_CARD, 2.0)]
            )

    def test_rejects_constant_empty_and_non_float_labels(self) -> None:
        calibration = HuberRegressionCalibration()
        with pytest.raises(ValueError, match="zero standard deviation"):
            calibration.calibrate(HuberLoss(), [(_CARD, 1.0), (_CARD, 1.0)])
        with pytest.raises(ValueError):
            calibration.calibrate(HuberLoss(), [])
        with pytest.raises(TypeError, match="float labels"):
            calibration.calibrate(HuberLoss(), [(_CARD, 1)])  # type: ignore[list-item]
