import math
from typing import Any

import pytest
import torch

from src.dojos.loss.bce_loss import BceLoss
from src.dojos.loss.fixed_classification_loss import FixedClassificationLoss
from src.dojos.loss.masked_vector_regression_loss import MaskedVectorRegressionLoss
from src.dojos.loss.pick_prediction_cross_entropy_loss import (
    PickPredictionCrossEntropyLoss,
)
from src.dojos.loss.prior_baseline_calibrations import (
    BinaryPriorCalibration,
    ClassPriorCalibration,
    MaskedVectorMeanCalibration,
    SoftTargetPriorCalibration,
    UniformOptionCalibration,
)
from src.dojos.loss.soft_classification_loss import SoftClassificationLoss

_CARD: Any = object()  # inputs are never read except by UniformOptionCalibration


def _sample(labels: list[Any]) -> list[tuple[Any, Any]]:
    return [(_CARD, label) for label in labels]


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class TestCalibrate:
    def test_keeps_the_loss_and_measures_the_baseline(self) -> None:
        loss = FixedClassificationLoss(["a", "b"])
        calibrated = ClassPriorCalibration().calibrate(loss, _sample(["a", "b"]))
        assert calibrated.loss is loss
        assert math.isclose(calibrated.baseline_loss, math.log(2))
        assert calibrated.label_stats is None

    def test_an_empty_sample_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ClassPriorCalibration().calibrate(FixedClassificationLoss(["a"]), [])

    def test_a_single_class_sample_has_a_zero_baseline_and_is_refused(self) -> None:
        with pytest.raises(ValueError, match="baseline_loss"):
            ClassPriorCalibration().calibrate(
                FixedClassificationLoss(["a", "b"]), _sample(["a", "a"])
            )


class TestClassPriorCalibration:
    def test_entropy_of_the_class_frequencies(self) -> None:
        baseline = ClassPriorCalibration().baseline_of(_sample(["a", "a", "a", "b"]))
        expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
        assert math.isclose(baseline, expected)

    def test_matches_the_loss_of_the_prior_predictor(self) -> None:
        labels = ["a", "a", "a", "b"]
        prior_logits = torch.log(torch.tensor([[0.75, 0.25]] * len(labels)))
        loss = FixedClassificationLoss(["a", "b"]).calculate(prior_logits, labels)
        baseline = ClassPriorCalibration().baseline_of(_sample(labels))
        assert math.isclose(loss.item(), baseline, rel_tol=1e-5)

    def test_refuses_a_non_string_label(self) -> None:
        with pytest.raises(TypeError, match="str labels"):
            ClassPriorCalibration().baseline_of(_sample([{"a": 1.0}]))


class TestBinaryPriorCalibration:
    def test_balanced_labels_give_ln_2(self) -> None:
        baseline = BinaryPriorCalibration().baseline_of(_sample([1.0, 0.0]))
        assert math.isclose(baseline, math.log(2))

    def test_matches_bce_of_the_positive_rate_predictor(self) -> None:
        labels = [1.0, 0.0, 0.0, 0.0]
        logits = torch.full((len(labels),), _logit(0.25))
        loss = BceLoss().calculate(logits, labels)  # type: ignore[arg-type]
        baseline = BinaryPriorCalibration().baseline_of(_sample(labels))
        assert math.isclose(loss.item(), baseline, rel_tol=1e-5)

    def test_refuses_labels_outside_zero_one(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            BinaryPriorCalibration().baseline_of(_sample([2.0]))

    def test_refuses_a_non_float_label(self) -> None:
        with pytest.raises(TypeError):
            BinaryPriorCalibration().baseline_of(_sample(["yes"]))


class TestSoftTargetPriorCalibration:
    def test_entropy_of_the_mean_target(self) -> None:
        baseline = SoftTargetPriorCalibration().baseline_of(
            _sample([{"a": 1.0}, {"b": 1.0}])
        )
        assert math.isclose(baseline, math.log(2))

    def test_matches_soft_cross_entropy_of_the_mean_target(self) -> None:
        labels = [{"a": 0.6, "b": 0.4}, {"a": 0.2, "b": 0.8}]
        mean_logits = torch.log(torch.tensor([[0.4, 0.6]] * len(labels)))
        loss = SoftClassificationLoss(["a", "b"]).calculate(mean_logits, labels)
        baseline = SoftTargetPriorCalibration().baseline_of(_sample(labels))
        assert math.isclose(loss.item(), baseline, rel_tol=1e-5)

    def test_refuses_masked_vector_labels(self) -> None:
        with pytest.raises(TypeError, match="str label keys"):
            SoftTargetPriorCalibration().baseline_of(_sample([{0: 0.5}]))


class TestMaskedVectorMeanCalibration:
    def test_masked_mse_of_the_position_means(self) -> None:
        baseline = MaskedVectorMeanCalibration().baseline_of(
            _sample([{0: 0.2}, {0: 0.6}])
        )
        assert math.isclose(baseline, 0.04)

    def test_matches_the_loss_of_the_position_mean_predictor(self) -> None:
        labels = [{0: 0.2, 1: 0.9}, {0: 0.6}, {1: 0.5, 2: 0.3}]
        means = [0.4, 0.7, 0.3]  # each position over the labels observing it
        logits = torch.tensor([[_logit(mean) for mean in means]] * len(labels))
        loss = MaskedVectorRegressionLoss(["0", "1", "2"]).calculate(
            logits, labels  # type: ignore[arg-type]
        )
        baseline = MaskedVectorMeanCalibration().baseline_of(_sample(labels))
        assert math.isclose(loss.item(), baseline, rel_tol=1e-5)

    def test_refuses_soft_target_labels(self) -> None:
        with pytest.raises(TypeError, match="int label keys"):
            MaskedVectorMeanCalibration().baseline_of(_sample([{"a": 0.5}]))

    def test_refuses_an_empty_label(self) -> None:
        with pytest.raises(ValueError, match="observe"):
            MaskedVectorMeanCalibration().baseline_of(_sample([{}]))


class TestUniformOptionCalibration:
    def test_mean_log_option_count(self) -> None:
        sample = [(["a", "b"], 0), (["a", "b", "c", "d"], 2)]
        baseline = UniformOptionCalibration(len).baseline_of(sample)
        assert math.isclose(baseline, (math.log(2) + math.log(4)) / 2)

    def test_matches_the_loss_of_constant_logits(self) -> None:
        sample = [(["a", "b"], 0), (["a", "b", "c"], 2)]
        logits = [torch.zeros(2), torch.zeros(3)]
        loss = PickPredictionCrossEntropyLoss().calculate(logits, [0, 2])
        baseline = UniformOptionCalibration(len).baseline_of(sample)
        assert math.isclose(loss.item(), baseline, rel_tol=1e-5)

    def test_refuses_an_input_without_options(self) -> None:
        with pytest.raises(ValueError, match="option"):
            UniformOptionCalibration(len).baseline_of([([], 0)])
