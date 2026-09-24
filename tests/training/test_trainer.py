import logging
from pathlib import Path

import pytest
import torch

from src.dojos.dojo import Dojo
from src.schema.holdout import HoldoutSpec
from src.training.plan import FaultPolicy, HardwareLimits, Precision
from src.training.recording.checkpointer import DirectoryCheckpointer
from src.training.recording.reports import CheckpointRecord, RoundReport
from src.training.trainer import Trainer
from tests.training.fakes import (
    LIMITS,
    FakeDojo,
    FakeModel,
    make_phase,
    make_plan,
)


class _RecordingListener:
    def __init__(self) -> None:
        self.reports: list[RoundReport] = []
        self.checkpoints: list[CheckpointRecord] = []

    def on_round_end(self, report: RoundReport) -> None:
        self.reports.append(report)

    def on_checkpoint(self, record: CheckpointRecord) -> None:
        self.checkpoints.append(record)


class _ExplodingListener:
    def on_round_end(self, report: RoundReport) -> None:
        raise RuntimeError("listener bug")

    def on_checkpoint(self, record: CheckpointRecord) -> None:
        raise RuntimeError("listener bug")


class _ExplodingCheckpointer:
    def save(self, *args: object, **kwargs: object) -> CheckpointRecord:
        raise OSError("disk full")


def _trainer(
    tmp_path: Path,
    dojos: list[FakeDojo],
    phase_names: tuple[str, ...] | None = None,
    listeners: list[object] | None = None,
    checkpointer: object | None = None,
    model: FakeModel | None = None,
    plan_overrides: dict[str, object] | None = None,
    precision: Precision = "fp32",
    **phase_overrides: object,
) -> Trainer:
    names = phase_names or tuple(d.name for d in dojos)
    plan = make_plan((make_phase(names, **phase_overrides),), **(plan_overrides or {}))
    return Trainer(
        model or FakeModel(),  # type: ignore[arg-type]
        dojos,  # type: ignore[arg-type]
        plan,
        HardwareLimits(LIMITS.max_batch_cost, precision),
        checkpointer or DirectoryCheckpointer(tmp_path),  # type: ignore[arg-type]
        listeners or [],  # type: ignore[arg-type]
    )


class TestRun:
    def test_trains_reports_each_round_and_keeps_the_best_checkpoint(
        self, tmp_path: Path
    ) -> None:
        listener = _RecordingListener()
        model = FakeModel()
        before = model.layer.weight.detach().clone()
        trainer = _trainer(
            tmp_path, [FakeDojo("a"), FakeDojo("b")], listeners=[listener], model=model
        )

        result = trainer.run()

        assert not torch.equal(before, model.layer.weight)
        assert result.stopped_early_reason is None
        assert result.final_report is not None
        assert result.final_report.step == 3 * 4
        assert set(result.final_report.per_dojo_test_loss) == {"a", "b"}
        assert result.best_checkpoint is not None
        assert result.best_checkpoint.encoder_path.exists()
        assert listener.checkpoints and len(listener.reports) >= 1

    def test_stops_the_phase_when_saturated(self, tmp_path: Path) -> None:
        # A dojo with no head and a frozen-free encoder still trains; zero lr
        # makes the loss constant so it saturates after patience_rounds.
        trainer = _trainer(
            tmp_path, [FakeDojo("a")], encoder_lr=0.0, head_lr=0.0, max_rounds=10
        )
        result = trainer.run()
        assert result.final_report is not None
        assert result.final_report.round_index < 9

    def test_a_frozen_encoder_is_not_updated_but_the_head_is(
        self, tmp_path: Path
    ) -> None:
        model = FakeModel()
        dojo = FakeDojo("a")
        encoder_before = model.layer.weight.detach().clone()
        head_before = [p.detach().clone() for p in dojo.trainable_parameters()]

        _trainer(tmp_path, [dojo], model=model, encoder_trainable=False).run()

        assert torch.equal(encoder_before, model.layer.weight)
        assert any(
            not torch.equal(b, p)
            for b, p in zip(head_before, dojo.trainable_parameters())
        )
        assert all(p.requires_grad is False for p in model.parameters())

    def test_a_later_phase_can_unfreeze_the_encoder(self, tmp_path: Path) -> None:
        model = FakeModel()
        dojo = FakeDojo("a")
        plan = make_plan(
            (
                make_phase(
                    ("a",), name="head_only", encoder_trainable=False, max_rounds=1
                ),
                make_phase(("a",), name="joint", max_rounds=1),
            )
        )
        checkpointer = DirectoryCheckpointer(tmp_path)
        Trainer(model, [dojo], plan, LIMITS, checkpointer, []).run()  # type: ignore[arg-type,list-item]  # noqa: E501
        assert all(p.requires_grad for p in model.parameters())

    def test_a_failing_dojo_is_quarantined_and_training_continues(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        dojos = [FakeDojo("good"), FakeDojo("bad", fail="train")]
        trainer = _trainer(
            tmp_path, dojos, plan_overrides={"faults": FaultPolicy(2, 1000)}
        )

        with caplog.at_level(logging.WARNING):
            result = trainer.run()

        assert result.final_report is not None
        assert "bad" in result.final_report.quarantined
        assert result.stopped_early_reason is None
        assert "quarantining" in caplog.text

    def test_a_nan_loss_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        dojos = [FakeDojo("good"), FakeDojo("nan", nan_loss=True)]
        model = FakeModel()
        result = _trainer(
            tmp_path,
            dojos,
            model=model,
            plan_overrides={"faults": FaultPolicy(2, 1000)},
        ).run()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert result.final_report is not None
        assert "nan" in result.final_report.quarantined

    def test_stops_cleanly_when_everything_keeps_failing(self, tmp_path: Path) -> None:
        dojos = [FakeDojo("bad", fail="train")]
        result = _trainer(
            tmp_path, dojos, plan_overrides={"faults": FaultPolicy(100, 3)}
        ).run()
        assert result.stopped_early_reason is not None
        assert result.best_checkpoint is None and result.final_report is None

    def test_ends_the_phase_when_every_dojo_is_quarantined(
        self, tmp_path: Path
    ) -> None:
        dojos = [FakeDojo("bad", fail="train")]
        result = _trainer(
            tmp_path, dojos, plan_overrides={"faults": FaultPolicy(2, 1000)}
        ).run()
        assert result.stopped_early_reason is not None
        assert "quarantined" in result.stopped_early_reason

    def test_survives_a_failing_checkpointer_and_listener(self, tmp_path: Path) -> None:
        result = _trainer(
            tmp_path,
            [FakeDojo("a")],
            listeners=[_ExplodingListener()],
            checkpointer=_ExplodingCheckpointer(),
        ).run()
        assert result.final_report is not None
        assert result.best_checkpoint is None

    def test_held_out_dojos_are_evaluated_but_not_trained(self, tmp_path: Path) -> None:
        held_out = FakeDojo("held_out", fail="train")
        result = _trainer(
            tmp_path,
            [FakeDojo("a"), held_out],
            phase_names=("a",),
            plan_overrides={"held_out_dojos": frozenset({"held_out"})},
        ).run()
        assert result.final_report is not None
        assert "held_out" in result.final_report.per_dojo_test_loss
        assert result.stopped_early_reason is None

    def test_the_run_is_reproducible_for_a_seed(self, tmp_path: Path) -> None:
        def final_weights() -> torch.Tensor:
            torch.manual_seed(0)
            model = FakeModel()
            _trainer(tmp_path / "x", [FakeDojo("a"), FakeDojo("b")], model=model).run()
            return model.layer.weight.detach().clone()

        # Batches come from torch.randn, so seed it identically each run
        assert torch.equal(final_weights(), final_weights())


class TestConstructorValidation:
    def test_rejects_a_dojo_with_a_different_holdout(self, tmp_path: Path) -> None:
        dojo = FakeDojo("a")
        dojo.holdout = HoldoutSpec(
            seed=1, tier_ratios=(8, 1, 1), held_out_games=frozenset()
        )
        with pytest.raises(ValueError, match="holdout"):
            _trainer(tmp_path, [dojo])

    def test_rejects_unregistered_dojo_names(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unregistered"):
            _trainer(tmp_path, [FakeDojo("a")], phase_names=("a", "ghost"))

    def test_rejects_duplicate_dojo_names(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unique"):
            _trainer(tmp_path, [FakeDojo("a"), FakeDojo("a")], phase_names=("a",))

    def test_rejects_a_phase_with_nothing_to_train(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no trainable"):
            _trainer(
                tmp_path,
                [FakeDojo("a", with_head=False)],
                encoder_trainable=False,
            )


class TestFailureRecovery:
    def test_a_failing_example_count_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        dojos = [FakeDojo("good"), FakeDojo("bad_count", count_fails=True)]
        result = _trainer(
            tmp_path, dojos, plan_overrides={"faults": FaultPolicy(2, 1000)}
        ).run()
        assert result.final_report is not None
        assert result.final_report.step > 0

    def test_a_finite_loss_with_nan_gradients_never_updates_the_weights(
        self, tmp_path: Path
    ) -> None:
        model = FakeModel()
        dojo = FakeDojo("a", nan_grad=True)
        _trainer(
            tmp_path,
            [dojo],
            model=model,
            plan_overrides={"faults": FaultPolicy(2, 1000)},
        ).run()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert all(torch.isfinite(p).all() for p in dojo.trainable_parameters())

    def test_a_dojo_failing_mid_pass_is_quarantined_by_total_failures(
        self, tmp_path: Path
    ) -> None:
        # Succeeds between failures, so only the total-failure limit catches it
        dojos = [FakeDojo("flaky", n_batches=3, fail_at_batch=2), FakeDojo("good")]
        policy = FaultPolicy(100, 1000, max_total_dojo_failures=2)
        result = _trainer(
            tmp_path, dojos, steps_per_round=12, plan_overrides={"faults": policy}
        ).run()
        assert result.final_report is not None
        assert "flaky" in result.final_report.quarantined

    def test_keeps_the_earlier_phases_checkpoint_when_a_later_phase_gives_up(
        self, tmp_path: Path
    ) -> None:
        plan = make_plan(
            (
                make_phase(("good",), name="first", max_rounds=1),
                make_phase(("bad",), name="second"),
            ),
            faults=FaultPolicy(100, 3),
        )
        dojos = [FakeDojo("good"), FakeDojo("bad", fail="train")]
        checkpointer = DirectoryCheckpointer(tmp_path)
        trainer = Trainer(FakeModel(), dojos, plan, LIMITS, checkpointer, [])  # type: ignore[arg-type]  # noqa: E501
        result = trainer.run()
        assert result.stopped_early_reason is not None
        assert result.best_checkpoint is not None
        assert result.best_checkpoint.report.phase == "first"

    def test_a_quarantined_dojo_does_not_block_the_phase_from_finishing(
        self, tmp_path: Path
    ) -> None:
        dojos = [FakeDojo("good"), FakeDojo("bad", fail="train")]
        result = _trainer(
            tmp_path,
            dojos,
            encoder_lr=0.0,
            head_lr=0.0,
            max_rounds=20,
            plan_overrides={"faults": FaultPolicy(1, 1000)},
        ).run()
        assert result.final_report is not None
        assert result.final_report.round_index < 19


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])))


class TestMixedPrecision:
    """fp16/bf16 autocast on CPU, which exercises the same GradScaler path
    as a GPU."""

    @pytest.mark.parametrize("precision", ["fp16", "bf16"])
    def test_trains_with_finite_weights_and_losses(
        self, tmp_path: Path, precision: Precision
    ) -> None:
        model = FakeModel()
        before = model.layer.weight.detach().clone()
        result = _trainer(
            tmp_path, [FakeDojo("a"), FakeDojo("b")], model=model, precision=precision
        ).run()

        assert result.stopped_early_reason is None
        assert not torch.equal(before, model.layer.weight)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert result.final_report is not None
        assert result.final_report.step > 0
        losses = result.final_report.per_dojo_test_loss.values()
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)

    def test_clips_unscaled_gradients(self, tmp_path: Path) -> None:
        # Clipping before unscaling would leave a norm 65536x too small
        model = FakeModel()
        dojo = FakeDojo("a")
        _trainer(
            tmp_path,
            [dojo],
            model=model,
            precision="fp16",
            max_grad_norm=1e-3,
            steps_per_round=4,
            max_rounds=1,
        ).run()
        parameters = [*model.parameters(), *dojo.trainable_parameters()]
        assert _gradient_norm(parameters) == pytest.approx(1e-3, rel=1e-2)

    def test_overflow_above_the_minimum_scale_skips_without_a_fault(
        self, tmp_path: Path
    ) -> None:
        # Overflows fp16 until the loss scale backs off to ~4 (14 halvings);
        # counted as faults, 2 in a row would quarantine the only dojo
        model = FakeModel()
        before = model.layer.weight.detach().clone()
        result = _trainer(
            tmp_path,
            [FakeDojo("a", loss_gain=1e4)],
            model=model,
            precision="fp16",
            steps_per_round=30,
            max_rounds=1,
            plan_overrides={"faults": FaultPolicy(2, 2)},
        ).run()

        assert result.stopped_early_reason is None
        assert result.final_report is not None
        assert "a" not in result.final_report.quarantined
        assert 0 < result.final_report.step < 30
        assert not torch.equal(before, model.layer.weight)

    def test_nan_gradients_never_update_and_fault_at_the_minimum_scale(
        self, tmp_path: Path
    ) -> None:
        # 16 skips take the scale from 65536 to 1; then each step is a fault
        model = FakeModel()
        dojo = FakeDojo("a", nan_grad=True)
        before = [p.detach().clone() for p in model.parameters()]
        result = _trainer(
            tmp_path,
            [dojo],
            model=model,
            precision="fp16",
            steps_per_round=40,
            max_rounds=1,
            plan_overrides={"faults": FaultPolicy(2, 1000)},
        ).run()

        assert result.stopped_early_reason == "every dojo quarantined in 'joint'"
        assert result.final_report is not None
        assert result.final_report.step == 0
        assert all(torch.equal(b, p) for b, p in zip(before, model.parameters()))
        assert all(torch.isfinite(p).all() for p in dojo.trainable_parameters())

    def test_a_step_failing_after_unscale_does_not_poison_later_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without a scaler reset, every later unscale_ raises and the only
        # dojo is quarantined
        real_clip = torch.nn.utils.clip_grad_norm_
        calls = {"n": 0}

        def clip_failing_once(*args: object, **kwargs: object) -> torch.Tensor:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated failure after unscale_")
            return real_clip(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_failing_once)
        result = _trainer(
            tmp_path,
            [FakeDojo("a")],
            precision="fp16",
            steps_per_round=8,
            max_rounds=1,
            plan_overrides={"faults": FaultPolicy(2, 1000)},
        ).run()

        assert result.stopped_early_reason is None
        assert result.final_report is not None
        assert result.final_report.step > 0


def test_fake_dojo_satisfies_the_protocol() -> None:
    dojo: Dojo = FakeDojo("a")
    assert dojo.name == "a"
