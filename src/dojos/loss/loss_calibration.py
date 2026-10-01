"""Fitting a generic dojo's loss to its TRAIN split.

Once a GenericDojo has its split files, it hands its cell's loss and a
sample of TRAIN examples to the cell's LossCalibration, and gets back a
CalibratedLoss: the loss it will actually train with (regression: labels
z-scored) and its baseline, the loss an input-ignoring predictor gets on
this data. Normalized loss is loss / baseline, so 1.0 means "learned
nothing" for every dojo.

LossCalibration is a Strategy owned by each generic cell, paired with its
loss: the cell that picks MseLoss also picks StandardizedRegressionCalibration.
Prior-based calibrations (classification, binary, soft targets, masked
vectors, option picks) live in prior_baseline_calibrations.py.
"""

from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence

from src.dojos.dojo import require_usable_baseline
from src.dojos.loss.label_stats import LabelStats
from src.dojos.loss.nocab_loss import NocabLoss
from src.dojos.loss.standardized_label_loss import StandardizedLabelLoss
from src.schema.type_hints import TrainingDatum

# The mean predictor's MSE on z-scored TRAIN labels (population std): exactly 1
_STANDARDIZED_MEAN_PREDICTOR_MSE = 1.0


@dataclass(frozen=True)
class CalibratedLoss:
    """A generic dojo's loss after fitting it to the TRAIN split.

    loss: what compute_loss scores with (possibly the cell's loss wrapped,
        e.g. in StandardizedLabelLoss).
    baseline_loss: the loss of the best input-ignoring predictor on the
        TRAIN sample, in the same units as loss; finite and > 0.
    label_stats (derived, not stored): the TRAIN LabelStats when loss is a
        StandardizedLabelLoss (regression cells), else None - read off the
        loss itself, so stats can never disagree with what the loss uses.
    """

    loss: NocabLoss[Any, Any]
    baseline_loss: float

    def __post_init__(self) -> None:
        """Validate baseline_loss.

        Side effects: none.
        Exceptions: ValueError if baseline_loss is not finite or <= 0 (a
            zero baseline - e.g. every TRAIN label the same class - leaves
            normalized loss undefined; a construction error, like a zero
            label std).
        """
        require_usable_baseline(self.baseline_loss, "calibrated loss")

    @property
    def label_stats(self) -> LabelStats | None:
        """self.loss's LabelStats if it is a StandardizedLabelLoss, else None.

        Side effects: none. Exceptions: none.

        Example:
            >>> CalibratedLoss(StandardizedLabelLoss(MseLoss(), stats), 1.0).label_stats
            LabelStats(mean=2.0, std=1.0)
        """
        if isinstance(self.loss, StandardizedLabelLoss):
            return self.loss.label_stats
        return None


class LossCalibration(Protocol):
    """Strategy: fit a cell's loss to a TRAIN sample (see module docstring)."""

    def calibrate(
        self, loss: NocabLoss[Any, Any], train_sample: Sequence[TrainingDatum]
    ) -> CalibratedLoss:
        """Fit loss to the TRAIN sample.

        Inputs: loss (the cell's own NocabLoss), train_sample (unmodded
            (input, label) TRAIN examples, as the dojo's data constructor
            built them; non-empty).
        Output: CalibratedLoss.
        Side effects: none (the calibration holds no state between calls).
        Exceptions: ValueError if train_sample is empty or its labels give
            no usable baseline; TypeError/ValueError if its labels are not
            the shape this calibration expects (a cell wired to the wrong
            calibration fails at construction, not mid-run).
        """
        ...


@dataclass(frozen=True)
class StandardizedRegressionCalibration:
    """LossCalibration for scalar regression cells: z-score the labels.

    Covers SingleCardRegressionDojo, MultiCardRegressionDojo and
    MultiGroupRegressionDojo (and so every wrapper that subclasses them).
    The loss is wrapped in StandardizedLabelLoss; the baseline is 1.0, the
    mean predictor's standardized MSE on TRAIN.
    """

    def calibrate(
        self, loss: NocabLoss[Any, Any], train_sample: Sequence[TrainingDatum]
    ) -> CalibratedLoss:
        """See LossCalibration.calibrate.

        Inputs: loss (a regression NocabLoss over float labels, e.g.
            MseLoss), train_sample (labels must be floats).
        Output: CalibratedLoss(StandardizedLabelLoss(loss, stats), 1.0).
        Side effects: none.
        Exceptions: ValueError if train_sample is empty, a label is not a
            finite float, or the labels have zero std (LabelStats.from_labels).

        Example:
            >>> StandardizedRegressionCalibration().calibrate(
            ...     MseLoss(), [(card_a, 1.0), (card_b, 3.0)]).label_stats
            LabelStats(mean=2.0, std=1.0)
        """
        result: CalibratedLoss

        # Parse the TRAIN labels into stats (raises on empty / constant)
        labels = _float_labels_of(train_sample)
        stats = LabelStats.from_labels(labels)

        # Train on z-scored labels; the mean predictor then scores exactly 1.0
        result = CalibratedLoss(
            loss=StandardizedLabelLoss(loss, stats),
            baseline_loss=_STANDARDIZED_MEAN_PREDICTOR_MSE,
        )
        return result


def _float_labels_of(train_sample: Sequence[TrainingDatum]) -> List[float]:
    """The sample's labels, checked to be floats.

    Inputs: train_sample (Sequence[TrainingDatum]).
    Output: List[float], same order.
    Side effects: none.
    Exceptions: TypeError naming the first non-float label (a regression
        cell wired to a non-regression constructor).
    """
    result: List[float] = []
    for _, label in train_sample:
        if not isinstance(label, float):
            raise TypeError(
                f"regression calibration needs float labels, got {type(label).__name__}"
            )
        result.append(label)
    return result
