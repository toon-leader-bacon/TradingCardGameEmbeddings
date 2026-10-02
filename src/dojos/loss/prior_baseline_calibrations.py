"""LossCalibrations that keep the cell's loss and only measure a baseline.

For every non-regression generic cell the loss is already on a natural
scale (nats, or [0, 1] probabilities), so calibrating means only finding
the baseline: the loss of the best input-ignoring predictor, fit to the
TRAIN sample. One subclass per loss family; they differ only in that
formula (Template Method).

- ClassPriorCalibration (FixedClassificationLoss): entropy of the TRAIN
  class frequencies.
- BinaryPriorCalibration (BceLoss): binary entropy of the TRAIN positive
  rate.
- SoftTargetPriorCalibration (SoftClassificationLoss): entropy of the
  mean TRAIN target distribution.
- MaskedVectorMeanCalibration (MaskedVectorRegressionLoss): masked MSE of
  the per-position TRAIN mean.
- UniformOptionCalibration (PickPredictionCrossEntropyLoss): mean
  ln(option count).

All entropies are in nats (natural log), matching torch's cross-entropy.
"""

import math
import numbers
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Type

from src.dojos.loss.loss_calibration import CalibratedLoss
from src.dojos.loss.nocab_loss import NocabLoss
from src.schema.type_hints import TrainingDatum, TrainingInput


class PriorBaselineCalibration(ABC):
    """Template Method base: the loss is kept, the baseline is measured.

    Subclasses implement only baseline_of. Satisfies LossCalibration.
    """

    def calibrate(
        self, loss: NocabLoss[Any, Any], train_sample: Sequence[TrainingDatum]
    ) -> CalibratedLoss:
        """See LossCalibration.calibrate.

        Inputs: loss (kept unchanged), train_sample (non-empty).
        Output: CalibratedLoss(loss, self.baseline_of(train_sample)).
        Side effects: none.
        Exceptions: ValueError if train_sample is empty or the baseline is
            zero (CalibratedLoss rejects it); whatever baseline_of raises for
            a label of the wrong shape.

        Example:
            >>> ClassPriorCalibration().calibrate(
            ...     FixedClassificationLoss(["a", "b"]), [(c1, "a"), (c2, "b")]
            ... ).baseline_loss  # ln 2
            0.6931...
        """
        result: CalibratedLoss

        # Validate inputs
        if not train_sample:
            raise ValueError("cannot calibrate a loss on an empty TRAIN sample")

        # The loss is on its natural scale already; only the baseline is new
        result = CalibratedLoss(loss=loss, baseline_loss=self.baseline_of(train_sample))
        return result

    @abstractmethod
    def baseline_of(self, train_sample: Sequence[TrainingDatum]) -> float:
        """The best input-ignoring predictor's loss on train_sample.

        Inputs: train_sample (non-empty Sequence[TrainingDatum]).
        Output: float >= 0 (0 only for a degenerate sample; the caller
            rejects it).
        Side effects: none.
        Exceptions: TypeError/ValueError for labels of the wrong shape.
        """
        ...


@dataclass(frozen=True)
class ClassPriorCalibration(PriorBaselineCalibration):
    """Fixed-class cross-entropy: baseline = H(TRAIN class frequencies).

    Labels are hashable class labels (str). The constant predictor that
    outputs the TRAIN class prior scores exactly this entropy on TRAIN.
    """

    def baseline_of(self, train_sample: Sequence[TrainingDatum]) -> float:
        """-sum_k p_k ln p_k over the sample's class frequencies p_k.

        Inputs/Side effects: see PriorBaselineCalibration.baseline_of.
        Output: float in [0, ln(classes)].

        Exceptions: TypeError if a label is not a str (e.g. a soft-target
            dict: the cell was wired to the wrong calibration).

        Example:
            >>> ClassPriorCalibration().baseline_of([(c1, "a"), (c2, "b")])  # ln 2
            0.6931...
        """
        counts: Counter[str] = Counter()
        for _, label in train_sample:
            if not isinstance(label, str):
                raise TypeError(
                    f"class-prior calibration needs str labels, got "
                    f"{type(label).__name__}"
                )
            counts[label] += 1
        total = sum(counts.values())
        return _entropy(count / total for count in counts.values())


@dataclass(frozen=True)
class BinaryPriorCalibration(PriorBaselineCalibration):
    """BCE: baseline = binary entropy of the TRAIN positive rate."""

    def baseline_of(self, train_sample: Sequence[TrainingDatum]) -> float:
        """-(p ln p + (1-p) ln(1-p)), p the mean of the 0.0/1.0 labels.

        Inputs/Side effects: see PriorBaselineCalibration.baseline_of.
        Output: float in [0, ln 2].

        Example:
            >>> BinaryPriorCalibration().baseline_of([(d1, 1.0), (d2, 0.0)])  # ln 2
            0.6931...
        Exceptions: TypeError if a label is not a float; ValueError if one
            lies outside [0, 1].
        """
        labels: List[float] = []
        for _, label in train_sample:
            if not isinstance(label, float):
                raise TypeError(
                    f"binary calibration needs float labels, got {type(label).__name__}"
                )
            if not 0.0 <= label <= 1.0:
                raise ValueError(f"binary labels must lie in [0, 1], got {label!r}")
            labels.append(label)
        positive_rate = math.fsum(labels) / len(labels)
        return _entropy((positive_rate, 1.0 - positive_rate))


@dataclass(frozen=True)
class SoftTargetPriorCalibration(PriorBaselineCalibration):
    """Soft cross-entropy: baseline = H(mean TRAIN target distribution).

    The constant predictor q minimizing E[-sum_k y_k ln q_k] is q = mean(y),
    so its loss is the cross-entropy of mean(y) with itself.
    """

    def baseline_of(self, train_sample: Sequence[TrainingDatum]) -> float:
        """-sum_k m_k ln m_k, m = elementwise mean of the label dicts
        (a key absent from a dict counts as 0).

        Inputs/Side effects: see PriorBaselineCalibration.baseline_of.
        Output: float >= 0.
        Exceptions: TypeError if a label is not a Dict[str, float] (str
            keys checked, so masked-vector int-keyed labels are refused).

        Example:
            >>> SoftTargetPriorCalibration().baseline_of(
            ...     [(c1, {"a": 1.0}), (c2, {"b": 1.0})])  # mean (.5, .5): ln 2
            0.6931...
        """
        totals: Dict[str, float] = defaultdict(float)
        for _, label in train_sample:
            for key, probability in _checked_dict_label(label, str).items():
                totals[key] += probability
        return _entropy(total / len(train_sample) for total in totals.values())


@dataclass(frozen=True)
class MaskedVectorMeanCalibration(PriorBaselineCalibration):
    """Masked, sigmoid-squashed vector MSE: baseline = that loss for a
    predictor outputting each position's TRAIN mean.

    Labels are sparse {position: value in [0, 1]} dicts. Not z-scored: the
    loss sigmoids the head's output, so its targets must stay in [0, 1].
    Mirrors MaskedVectorRegressionLoss's weighting: each example's squared
    error is averaged over its own observed positions, then over examples.
    (The per-position mean is not exactly the minimizer under that
    weighting; close enough for a baseline, and deterministic.)
    """

    def baseline_of(self, train_sample: Sequence[TrainingDatum]) -> float:
        """mean over examples of mean over observed positions b of
        (y_b - mean_b)^2, mean_b the TRAIN mean of position b.

        Inputs/Side effects: see PriorBaselineCalibration.baseline_of.
        Output: float >= 0.
        Exceptions: TypeError if a label is not a Dict[int, float] (int
            keys checked, so soft-target str-keyed labels are refused);
            ValueError if a label dict is empty.

        Example:
            >>> MaskedVectorMeanCalibration().baseline_of(
            ...     [(c1, {0: 0.2}), (c2, {0: 0.6})])  # mean 0.4: 0.2^2
            0.04
        """
        labels = [_checked_dict_label(label, int) for _, label in train_sample]
        position_means = _observed_position_means(labels)
        per_example_errors: List[float] = []
        for label in labels:
            if not label:
                raise ValueError("a masked-vector label must observe a position")
            squared_errors = [
                (value - position_means[position]) ** 2
                for position, value in label.items()
            ]
            per_example_errors.append(math.fsum(squared_errors) / len(squared_errors))
        return math.fsum(per_example_errors) / len(per_example_errors)


@dataclass(frozen=True)
class UniformOptionCalibration(PriorBaselineCalibration):
    """Pick prediction over a ragged option list: baseline = mean ln(n).

    A constant-logit head spreads probability uniformly over an example's
    n options, scoring ln(n) whatever was picked.

    option_count: how many options an input offers. The option cells
        supply it, since they know their input shape: len(pack) for
        MultiCardOptionSelectionDojo, len(input[0]) (the pack group) for
        MultiGroupOptionSelectionDojo.
    """

    option_count: Callable[[TrainingInput], int]

    def baseline_of(self, train_sample: Sequence[TrainingDatum]) -> float:
        """mean over examples of ln(option_count(input)).

        Inputs/Side effects: see PriorBaselineCalibration.baseline_of.
        Output: float >= 0 (0 only if every example has one option).
        Exceptions: ValueError if an input has fewer than one option.

        Example:
            >>> UniformOptionCalibration(len).baseline_of(
            ...     [([a, b], 0), ([a, b, c, d], 2)])  # (ln 2 + ln 4) / 2
            1.0397...
        """
        log_counts: List[float] = []
        for training_input, _ in train_sample:
            count = self.option_count(training_input)
            if count < 1:
                raise ValueError(f"an option input needs >= 1 option, got {count}")
            log_counts.append(math.log(count))
        return math.fsum(log_counts) / len(log_counts)


def _entropy(probabilities: Iterable[float]) -> float:
    """Shannon entropy in nats: -sum p ln p, skipping p == 0 (0 ln 0 = 0).

    Inputs: probabilities (Iterable[float]), each in [0, 1].
    Output: float >= 0.
    Side effects: none. Exceptions: none.
    """
    return -math.fsum(p * math.log(p) for p in probabilities if p > 0.0)


def _checked_dict_label(label: Any, key_type: Type[Any]) -> Dict[Any, float]:
    """label, checked to be a dict whose keys are key_type and values numbers.

    Shared by the two dict-label calibrations, so each refuses the other's
    labels (str-keyed soft targets vs int-keyed masked vectors).
    Inputs: label (one example's label), key_type (str or int).
    Output: label as a Dict[key_type, float].
    Side effects: none.
    Exceptions: TypeError if label is not a dict, a key is not key_type (a
        bool is not accepted as an int), or a value is not a number.
    """
    if not isinstance(label, dict):
        raise TypeError(f"expected a dict label, got {type(label).__name__}")
    for key, value in label.items():
        if not isinstance(key, key_type) or isinstance(key, bool):
            raise TypeError(
                f"expected {key_type.__name__} label keys, got {type(key).__name__}"
            )
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f"expected numeric label values, got {value!r}")
    return label


def _observed_position_means(labels: Sequence[Mapping[int, float]]) -> Dict[int, float]:
    """Each position's mean value over the labels that observe it.

    Inputs: labels, sparse {position: value} dicts.
    Output: position -> mean of its observed values.
    Side effects: none. Exceptions: none.
    """
    totals: Dict[int, float] = defaultdict(float)
    counts: Counter[int] = Counter()
    for label in labels:
        for position, value in label.items():
            totals[position] += value
            counts[position] += 1
    return {position: totals[position] / counts[position] for position in totals}
