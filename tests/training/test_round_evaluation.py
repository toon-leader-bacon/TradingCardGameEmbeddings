import logging

import pytest

from src.encoder_model.precision import Precision
from src.schema.splits import Split
from src.training.round_evaluation import evaluate_split_losses
from tests.training.fakes import BUDGET, FakeDojo, FakeModel


def _score(
    model: FakeModel,
    dojos: list[FakeDojo],
    split: Split = Split.TEST,
    precision: Precision = "fp32",
) -> dict[str, float]:
    return dict(
        evaluate_split_losses(
            model,  # type: ignore[arg-type]
            dojos,  # type: ignore[arg-type]
            split=split,
            budget=BUDGET,
            max_examples=8,
            precision=precision,
        )
    )


def test_scores_every_dojo_by_name() -> None:
    dojos = [FakeDojo("a"), FakeDojo("b")]
    losses = _score(FakeModel(), dojos)
    assert set(losses) == {"a", "b"}
    assert all(loss >= 0 for loss in losses.values())


def test_a_failing_dojo_is_logged_and_omitted(caplog: pytest.LogCaptureFixture) -> None:
    dojos = [FakeDojo("a"), FakeDojo("bad", fail="test")]
    with caplog.at_level(logging.WARNING):
        losses = _score(FakeModel(), dojos)
    assert set(losses) == {"a"}
    assert "bad" in caplog.text


def test_a_non_finite_or_empty_dojo_is_omitted() -> None:
    dojos = [FakeDojo("nan", nan_loss=True), FakeDojo("empty", n_batches=0)]
    assert _score(FakeModel(), dojos) == {}


def test_restores_the_models_previous_mode_and_builds_no_graph() -> None:
    model = FakeModel()
    model.train(False)
    _score(model, [FakeDojo("a")])
    assert model.training is False
    model.train(True)
    _score(model, [FakeDojo("bad", fail="test")])
    assert model.training is True


def test_scores_the_requested_split_only() -> None:
    # This dojo cannot serve VALIDATION but serves TEST fine
    dojo = FakeDojo("a", fail="validation")
    assert set(_score(FakeModel(), [dojo], split=Split.TEST)) == {"a"}
    assert _score(FakeModel(), [dojo], split=Split.VALIDATION) == {}


def test_runs_under_autocast_at_a_lower_precision() -> None:
    # bf16 autocast is supported on CPU; losses must still come back finite
    losses = _score(FakeModel(), [FakeDojo("a")], precision="bf16")
    assert set(losses) == {"a"}
    assert losses["a"] == losses["a"]  # not NaN
