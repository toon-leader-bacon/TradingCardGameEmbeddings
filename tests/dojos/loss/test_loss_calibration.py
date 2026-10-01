import math
from typing import Any

import pytest
import torch

from src.dojos.loss.fixed_classification_loss import FixedClassificationLoss
from src.dojos.loss.label_stats import LabelStats
from src.dojos.loss.loss_calibration import (
    CalibratedLoss,
    StandardizedRegressionCalibration,
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
