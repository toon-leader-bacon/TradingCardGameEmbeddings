"""Shared fixtures for dojo tests."""

from typing import Any, Iterator

import pytest

from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.loss.loss_calibration import CalibratedLoss, LossCalibration
from src.dojos.loss.nocab_loss import NocabLoss


@pytest.fixture
def uncalibrated_generic_dojos(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Skip GenericDojo's TRAIN calibration pass: the cell's loss is kept
    as built and the baseline is 1.0.

    For wrapper tests that check only wiring (path, constructor, label
    values, mods) against a placeholder parquet whose columns the real
    data constructor could not read. Calibration itself is tested in
    tests/dojos/loss/ and tests/dojos/generic/test_generic_dojo.py.
    Side effects: patches GenericDojo._calibrated_loss for one test.
    """

    def keep_loss(
        self: GenericDojo, loss_calculator: NocabLoss[Any, Any], _: LossCalibration
    ) -> CalibratedLoss:
        return CalibratedLoss(loss=loss_calculator, baseline_loss=1.0)

    monkeypatch.setattr(GenericDojo, "_calibrated_loss", keep_loss)
    yield
