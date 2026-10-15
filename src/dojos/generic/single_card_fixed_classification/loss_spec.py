"""LossSpec: a single-card classification cell's loss and
the LossCalibration that goes with it, as one value.

SingleCardFixedClassificationDojo lets a wrapper swap its loss (soft
targets, masked vectors). The loss and its calibration must change
together - a SoftClassificationLoss measured with ClassPriorCalibration
would get a meaningless baseline - so a wrapper picks one spec, never the
two halves separately.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Sequence

from src.dojos.loss.fixed_classification_loss import FixedClassificationLoss
from src.dojos.loss.loss_calibration import LossCalibration
from src.dojos.loss.masked_vector_regression_loss import MaskedVectorRegressionLoss
from src.dojos.loss.nocab_loss import NocabLoss
from src.dojos.loss.prior_baseline_calibrations import (
    ClassPriorCalibration,
    MaskedVectorMeanCalibration,
    SoftTargetPriorCalibration,
)
from src.dojos.loss.soft_classification_loss import SoftClassificationLoss
from src.schema.type_hints import LabelT


@dataclass(frozen=True)
class LossSpec(Generic[LabelT]):
    """A matched (loss, calibration) pair for a classification-shaped head.

    loss_factory: builds the loss from the cell's label_values.
    calibration: fits that loss to the TRAIN split (its baseline).
    """

    loss_factory: Callable[[Sequence[str]], NocabLoss[Any, List[LabelT]]]
    calibration: LossCalibration[LabelT]


# Hard class labels: cross-entropy, baseline = TRAIN class-prior entropy
FIXED_CLASSIFICATION_LOSS_SPEC: LossSpec[str] = LossSpec(
    FixedClassificationLoss, ClassPriorCalibration()
)
# Probability-vector labels (CardCharacterPredictionDojo)
SOFT_CLASSIFICATION_LOSS_SPEC: LossSpec[Dict[str, float]] = LossSpec(
    SoftClassificationLoss, SoftTargetPriorCalibration()
)
# Sparse per-position [0, 1] labels (PickNumberDecayCurveDojo)
MASKED_VECTOR_REGRESSION_LOSS_SPEC: LossSpec[Dict[int, float]] = LossSpec(
    MaskedVectorRegressionLoss, MaskedVectorMeanCalibration()
)
