"""A regression cell's loss and its matching calibration, as one value.

The loss and the calibration that fits its baseline are a coupled pair:
the Huber loss needs the best-constant Huber baseline, and MseLoss needs
the mean predictor's 1.0. Handing the three regression cells one
RegressionObjective keeps the pair from being mismatched.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.dojos.loss.huber_loss import DEFAULT_HUBER_DELTA, HuberLoss
from src.dojos.loss.loss_calibration import (
    HuberRegressionCalibration,
    LossCalibration,
    StandardizedRegressionCalibration,
)
from src.dojos.loss.mse_loss import MseLoss
from src.dojos.loss.nocab_loss import NocabLoss


class RegressionLossKind(StrEnum):
    """Which loss the regression dojos of a run train with (the run
    config's `regression_loss:` value)."""

    MSE = "mse"
    HUBER = "huber"


@dataclass(frozen=True)
class RegressionObjective:
    """What a regression cell trains with: loss and calibration, matched.

    Build with RegressionObjective.huber() or .mse().

    Exceptions (constructor): ValueError if loss and calibration are not a
        matching pair (HuberLoss with HuberRegressionCalibration, or
        MseLoss with StandardizedRegressionCalibration).
    """

    loss: NocabLoss[Any, Any]
    calibration: LossCalibration

    def __post_init__(self) -> None:
        # Each loss type must come with its own calibration
        is_huber = isinstance(self.loss, HuberLoss) and isinstance(
            self.calibration, HuberRegressionCalibration
        )
        is_mse = isinstance(self.loss, MseLoss) and isinstance(
            self.calibration, StandardizedRegressionCalibration
        )
        if not (is_huber or is_mse):
            raise ValueError(
                f"{type(self.loss).__name__} does not match "
                f"{type(self.calibration).__name__}"
            )

    @classmethod
    def huber(cls, delta: float = DEFAULT_HUBER_DELTA) -> "RegressionObjective":
        """Huber loss with its best-constant baseline.

        Inputs: delta (float, finite and > 0). Output: RegressionObjective.
        Side effects: none. Exceptions: ValueError for a bad delta.

        Example:
            >>> RegressionObjective.huber(1.0).loss.delta
            1.0
        """
        return cls(HuberLoss(delta), HuberRegressionCalibration())

    @classmethod
    def mse(cls) -> "RegressionObjective":
        """Today's pair: MSE with the mean predictor's baseline of 1.0.

        Inputs: none. Output: RegressionObjective. Side effects: none.
        Exceptions: none.

        Example:
            >>> isinstance(RegressionObjective.mse().loss, MseLoss)
            True
        """
        return cls(MseLoss(), StandardizedRegressionCalibration())

    @classmethod
    def for_kind(cls, kind: RegressionLossKind) -> "RegressionObjective":
        """The objective a run config's `regression_loss:` names.

        Inputs: kind (RegressionLossKind). Output: RegressionObjective
        (Huber at DEFAULT_HUBER_DELTA for HUBER). Side effects: none.
        Exceptions: ValueError for a kind this method does not handle.

        Example:
            >>> RegressionObjective.for_kind(RegressionLossKind.HUBER).loss.delta
            1.345
        """
        if kind is RegressionLossKind.HUBER:
            return cls.huber()
        if kind is RegressionLossKind.MSE:
            return cls.mse()
        raise ValueError(f"no RegressionObjective for {kind!r}")
