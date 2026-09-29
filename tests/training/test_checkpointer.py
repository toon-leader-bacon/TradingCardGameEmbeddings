import json
from pathlib import Path

import pytest
import torch

from src.schema.holdout import HoldoutSpec
from src.training.recording.checkpointer import (
    DirectoryCheckpointer,
    load_checkpoint_holdout,
    load_encoder_weights,
)
from src.training.recording.reports import DojoStatus, RoundReport
from tests.training.fakes import FakeDojo, FakeModel, make_phase, make_plan


def _save(tmp_path: Path, phase: str = "pre train") -> tuple[Path, FakeModel]:
    model = FakeModel()
    dojo = FakeDojo("a")
    optimizer = torch.optim.AdamW(model.parameters())
    report = RoundReport(
        phase, 2, 8, 1.5, {"a": 0.5}, {"a": DojoStatus.ACTIVE}, frozenset()
    )
    plan = make_plan((make_phase(("a",)),))
    checkpointer = DirectoryCheckpointer(tmp_path)
    record = checkpointer.save(model, [dojo], optimizer, plan, report)  # type: ignore[list-item,arg-type]  # noqa: E501
    return record.path.parent, model


def test_writes_state_encoder_and_manifest(tmp_path: Path) -> None:
    directory, model = _save(tmp_path)
    assert {p.name for p in directory.iterdir()} == {
        "state.pt",
        "encoder.pt",
        "manifest.json",
        "holdout.json",
    }
    encoder = torch.load(directory / "encoder.pt", weights_only=True)
    assert set(encoder) == set(model.state_dict())
    state = torch.load(directory / "state.pt", weights_only=True)
    assert set(state) == {"model", "optimizer", "dojo_heads"}
    assert len(state["dojo_heads"]["a"]) == 2  # weight and bias
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["round_index"] == 2 and manifest["per_dojo_test_loss"] == {"a": 0.5}


def test_directory_name_is_filename_safe_and_no_staging_dir_remains(
    tmp_path: Path,
) -> None:
    directory, _ = _save(tmp_path, phase="pre train/1")
    assert directory.name == "pre_train_1_round0002"
    assert [p.name for p in tmp_path.iterdir()] == [directory.name]


def test_a_failed_write_leaves_no_partial_checkpoint(tmp_path: Path) -> None:
    class Broken(FakeModel):
        def encoder_only_state_dict(self):  # type: ignore[no-untyped-def]
            raise OSError("disk full")

    report = RoundReport("p", 0, 0, 0.0, {}, {}, frozenset())
    plan = make_plan((make_phase(("a",)),))
    model = Broken()
    try:
        DirectoryCheckpointer(tmp_path).save(
            model, [], torch.optim.AdamW(model.parameters()), plan, report  # type: ignore[arg-type]
        )
    except OSError:
        pass
    assert list(tmp_path.iterdir()) == []


def test_the_holdout_round_trips_through_the_checkpoint(tmp_path: Path) -> None:
    directory, _ = _save(tmp_path)
    assert load_checkpoint_holdout(directory) == HoldoutSpec.no_holdout()


def test_a_checkpoint_without_holdout_json_raises_file_not_found(
    tmp_path: Path,
) -> None:
    directory, _ = _save(tmp_path)
    (directory / "holdout.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_checkpoint_holdout(directory)


def test_encoder_weights_load_into_a_fresh_model(tmp_path: Path) -> None:
    directory, saved = _save(tmp_path)
    fresh = FakeModel()
    assert not torch.equal(fresh.layer.weight, saved.layer.weight)
    load_encoder_weights(directory, fresh)
    assert torch.equal(fresh.layer.weight, saved.layer.weight)
    assert torch.equal(fresh.layer.bias, saved.layer.bias)


def test_encoder_weights_reject_a_different_architecture(tmp_path: Path) -> None:
    directory, _ = _save(tmp_path)
    with pytest.raises(RuntimeError):
        load_encoder_weights(directory, torch.nn.Linear(5, 5))


def test_encoder_weights_missing_raise_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_encoder_weights(tmp_path, FakeModel())
