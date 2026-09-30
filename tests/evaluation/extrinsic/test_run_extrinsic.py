import copy
import json
import logging
from pathlib import Path

import pytest
import torch

import src.evaluation.extrinsic.run_extrinsic as module
from src.evaluation.extrinsic.run_extrinsic import ExtrinsicSpec, run_extrinsic
from src.schema.holdout import HoldoutSpec
from src.training.plan import Proportional
from src.training.recording.run_listener import read_rounds_csv
from tests.training.fakes import LIMITS, FakeDojo, FakeModel, saturation_spec


class _ResettableDojo(FakeDojo):
    """reset_head restores the as-built head, as GenericDojo's does."""

    def __init__(self, name: str, **kwargs: object) -> None:
        super().__init__(name, **kwargs)  # type: ignore[arg-type]
        self._initial_head = copy.deepcopy(self._head.state_dict())

    def reset_head(self) -> None:
        self._head.load_state_dict(self._initial_head)


def _spec(**overrides: object) -> ExtrinsicSpec:
    fields: dict = dict(
        diet_rule=Proportional(),
        head_lr=0.05,
        steps_per_round=4,
        max_rounds=3,
        saturation=saturation_spec(),
        eval_examples_per_dojo=8,
        seed=0,
    )
    fields.update(overrides)
    return ExtrinsicSpec(**fields)


def _run(  # type: ignore[no-untyped-def]
    encoder, dojos, tmp_path: Path, label: str = "enc", **kwargs
):
    return run_extrinsic(encoder, label, dojos, _spec(**kwargs), LIMITS, tmp_path)


def test_validation_scores_the_heads_restored_to_their_best_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Record the order of the restore and the VALIDATION pass, and the head
    # the pass sees: it must be the one the keeper restored
    calls: list[str] = []
    seen_heads: list[list[torch.Tensor]] = []
    restore = module.BestHeadKeeper.restore_best_heads

    def recording_restore(self):  # type: ignore[no-untyped-def]
        calls.append("restore")
        restore(self)
        seen_heads.append([p.detach().clone() for p in self._snapshots["a"]])

    def recording_evaluate(encoder, dojos, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("validation")
        head = [p.detach().clone() for p in dojos[0].trainable_parameters()]
        assert all(torch.equal(x, y) for x, y in zip(head, seen_heads[0]))
        return {dojo.name: 1.0 for dojo in dojos}

    monkeypatch.setattr(module.BestHeadKeeper, "restore_best_heads", recording_restore)
    monkeypatch.setattr(module, "evaluate_split_losses", recording_evaluate)

    result = _run(FakeModel(), [FakeDojo("a")], tmp_path)

    assert calls == ["restore", "validation"]
    assert result.validation_losses == {"a": 1.0}


def test_a_run_trains_the_heads_only_and_records_both_passes(tmp_path: Path) -> None:
    encoder = FakeModel()
    dojos = [FakeDojo("a"), FakeDojo("b")]
    encoder_before = encoder.layer.weight.detach().clone()
    head_before = [p.detach().clone() for p in dojos[0].trainable_parameters()]

    result = _run(encoder, dojos, tmp_path)

    run_dir = tmp_path / "extrinsic" / "enc"
    assert result.rounds_csv == run_dir / "rounds.csv"
    rows = read_rounds_csv(run_dir / "rounds.csv")
    assert {row.phase for row in rows} == {"extrinsic"}
    assert {row.dojo for row in rows} == {"a", "b"}
    assert result.stopped_early_reason is None and result.quarantined == frozenset()
    assert set(result.validation_losses or {}) == {"a", "b"}
    written = json.loads((run_dir / "validation.json").read_text())
    assert written["losses"] == dict(result.validation_losses or {})
    assert written["quarantined"] == []
    # The scored head of each dojo is the one from its best TEST round
    best_rows = {
        dojo: min((row for row in rows if row.dojo == dojo), key=lambda r: r.test_loss)
        for dojo in ("a", "b")
    }
    assert written["best_test_rounds"] == {
        dojo: row.round_index for dojo, row in best_rows.items()
    }
    # The encoder is frozen and untouched; the heads trained
    assert torch.equal(encoder.layer.weight, encoder_before)
    assert all(not p.requires_grad for p in encoder.parameters())
    assert encoder.training is False
    assert any(
        not torch.equal(before, after)
        for before, after in zip(head_before, dojos[0].trainable_parameters())
    )


def test_reused_dojos_give_identical_encoders_identical_runs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    first = FakeModel()
    second = copy.deepcopy(first)
    dojos = [_ResettableDojo("a"), _ResettableDojo("b")]

    first_result = _run(first, dojos, tmp_path, label="first")
    second_result = _run(second, dojos, tmp_path, label="second")

    # Without reset_head the second run would start from trained heads
    def losses(label: str) -> list[float]:
        rows = read_rounds_csv(tmp_path / "extrinsic" / label / "rounds.csv")
        return [row.test_loss for row in rows]

    assert losses("first") == losses("second")
    assert first_result.validation_losses == second_result.validation_losses


def test_the_result_losses_are_read_only(tmp_path: Path) -> None:
    result = _run(FakeModel(), [FakeDojo("a")], tmp_path)
    with pytest.raises(TypeError):
        result.validation_losses["a"] = 0.0  # type: ignore[index]


def test_a_dojo_failing_validation_is_omitted(tmp_path: Path) -> None:
    result = _run(
        FakeModel(), [FakeDojo("a"), FakeDojo("b", fail="validation")], tmp_path
    )
    assert set(result.validation_losses or {}) == {"a"}


@pytest.mark.parametrize("label", ["", ".", "..", "a/b", "/abs", "a\0b"])
def test_a_bad_label_is_refused_before_anything_is_written(
    tmp_path: Path, label: str
) -> None:
    with pytest.raises(ValueError, match="path segment"):
        _run(FakeModel(), [FakeDojo("a")], tmp_path, label=label)
    assert list(tmp_path.iterdir()) == []


def _other_holdout_dojo() -> FakeDojo:
    dojo = FakeDojo("b")
    dojo.holdout = HoldoutSpec(
        seed=9, tier_ratios=(8, 1, 1), held_out_games=frozenset()
    )
    return dojo


@pytest.mark.parametrize(
    ("dojos", "spec_overrides"),
    [
        ([], {}),  # no dojos
        ([FakeDojo("a", with_head=False)], {}),  # nothing to train
        ([FakeDojo("a")], {"steps_per_round": 0}),  # Phase rejects
        ([FakeDojo("a")], {"eval_examples_per_dojo": 0}),  # TrainingPlan rejects
        ([FakeDojo("a"), FakeDojo("a")], {}),  # Trainer: duplicate names
        ([FakeDojo("a"), _other_holdout_dojo()], {}),  # Trainer: holdouts differ
    ],
)
def test_invalid_runs_are_refused_before_anything_is_written(
    tmp_path: Path, dojos: list, spec_overrides: dict
) -> None:
    with pytest.raises(ValueError):
        _run(FakeModel(), dojos, tmp_path, **spec_overrides)
    assert list(tmp_path.iterdir()) == []


def test_an_existing_run_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "extrinsic" / "enc").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        _run(FakeModel(), [FakeDojo("a")], tmp_path)


def test_a_quarantined_dojo_is_recorded_and_the_run_completes(tmp_path: Path) -> None:
    dojos = [FakeDojo("a"), FakeDojo("b", fail="train")]
    result = _run(FakeModel(), dojos, tmp_path, steps_per_round=12)
    assert result.stopped_early_reason is None
    assert result.quarantined == frozenset({"b"})
    written = json.loads(
        (tmp_path / "extrinsic" / "enc" / "validation.json").read_text()
    )
    assert written["quarantined"] == ["b"]


def test_a_run_that_stops_early_skips_validation(tmp_path: Path) -> None:
    result = _run(FakeModel(), [FakeDojo("a", fail="train")], tmp_path)
    assert result.stopped_early_reason is not None
    assert result.validation_losses is None
    assert not (tmp_path / "extrinsic" / "enc" / "validation.json").exists()


def test_a_failed_validation_pass_is_logged_and_the_run_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_pass(*args: object, **kwargs: object) -> None:
        raise RuntimeError("autocast unsupported")

    monkeypatch.setattr(module, "evaluate_split_losses", failing_pass)
    with caplog.at_level(logging.ERROR):
        result = _run(FakeModel(), [FakeDojo("a")], tmp_path)
    assert result.validation_losses is None
    assert result.rounds_csv is not None
    assert not (tmp_path / "extrinsic" / "enc" / "validation.json").exists()
    assert "VALIDATION pass failed" in caplog.text


def test_a_failed_validation_write_is_logged_and_the_losses_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original = Path.write_text

    def failing_write(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == "validation.json":
            raise OSError("disk full")
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", failing_write)
    with caplog.at_level(logging.ERROR):
        result = _run(FakeModel(), [FakeDojo("a")], tmp_path)
    assert set(result.validation_losses or {}) == {"a"}
    assert "could not write" in caplog.text
