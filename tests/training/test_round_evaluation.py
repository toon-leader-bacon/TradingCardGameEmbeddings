import logging

import pytest

from src.training.round_evaluation import evaluate_test_losses
from tests.training.fakes import BUDGET, FakeDojo, FakeModel


def test_scores_every_dojo_by_name() -> None:
    dojos = [FakeDojo("a"), FakeDojo("b")]
    losses = evaluate_test_losses(FakeModel(), dojos, BUDGET, 8)  # type: ignore[arg-type]
    assert set(losses) == {"a", "b"}
    assert all(loss >= 0 for loss in losses.values())


def test_a_failing_dojo_is_logged_and_omitted(caplog: pytest.LogCaptureFixture) -> None:
    dojos = [FakeDojo("a"), FakeDojo("bad", fail="test")]
    with caplog.at_level(logging.WARNING):
        losses = evaluate_test_losses(FakeModel(), dojos, BUDGET, 8)  # type: ignore[arg-type]
    assert set(losses) == {"a"}
    assert "bad" in caplog.text


def test_a_non_finite_or_empty_dojo_is_omitted() -> None:
    dojos = [FakeDojo("nan", nan_loss=True), FakeDojo("empty", n_batches=0)]
    assert evaluate_test_losses(FakeModel(), dojos, BUDGET, 8) == {}  # type: ignore[arg-type]


def test_restores_the_models_previous_mode_and_builds_no_graph() -> None:
    model = FakeModel()
    model.train(False)
    evaluate_test_losses(model, [FakeDojo("a")], BUDGET, 8)  # type: ignore[arg-type]
    assert model.training is False
    model.train(True)
    evaluate_test_losses(model, [FakeDojo("bad", fail="test")], BUDGET, 8)  # type: ignore[arg-type]
    assert model.training is True
