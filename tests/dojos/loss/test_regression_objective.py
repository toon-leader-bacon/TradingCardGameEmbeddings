import pytest

from src.dojos.loss.huber_loss import HuberLoss
from src.dojos.loss.loss_calibration import (
    HuberRegressionCalibration,
    StandardizedRegressionCalibration,
)
from src.dojos.loss.mse_loss import MseLoss
from src.dojos.loss.regression_objective import (
    RegressionLossKind,
    RegressionObjective,
)


class TestFactories:
    def test_mse_is_todays_pair(self) -> None:
        objective = RegressionObjective.mse()
        assert isinstance(objective.loss, MseLoss)
        assert isinstance(objective.calibration, StandardizedRegressionCalibration)

    def test_huber_is_the_huber_pair_with_the_given_delta(self) -> None:
        objective = RegressionObjective.huber(delta=2.0)
        assert isinstance(objective.loss, HuberLoss)
        assert objective.loss.delta == 2.0
        assert isinstance(objective.calibration, HuberRegressionCalibration)

    def test_huber_rejects_a_bad_delta(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            RegressionObjective.huber(delta=0.0)

    def test_for_kind_maps_each_kind(self) -> None:
        huber = RegressionObjective.for_kind(RegressionLossKind.HUBER)
        mse = RegressionObjective.for_kind(RegressionLossKind.MSE)
        assert isinstance(huber.loss, HuberLoss)
        assert isinstance(mse.loss, MseLoss)

    def test_kinds_are_parsed_from_their_config_names(self) -> None:
        assert RegressionLossKind("huber") is RegressionLossKind.HUBER
        assert RegressionLossKind("mse") is RegressionLossKind.MSE


class TestMismatchedPairs:
    def test_rejects_huber_loss_with_the_mse_calibration(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            RegressionObjective(HuberLoss(), StandardizedRegressionCalibration())

    def test_rejects_mse_loss_with_the_huber_calibration(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            RegressionObjective(MseLoss(), HuberRegressionCalibration())
