import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch
import yaml

from src.training.run_config import EmbeddingHeadKind, ModelSpec

_SCRIPT = Path("scripts/run_training.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_training_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("head", list(EmbeddingHeadKind))
def test_every_embedding_head_kind_builds(head: EmbeddingHeadKind) -> None:
    script = _load_script()
    built = script.build_embedding_head(ModelSpec(head, "ckpt", 16), input_dim=8)
    assert built.output_dim == 16


def test_the_residual_mlp_head_takes_its_sizes() -> None:
    script = _load_script()
    spec = ModelSpec(EmbeddingHeadKind.RESIDUAL_MLP, "ckpt", 16, 32, 5)
    built = script.build_embedding_head(spec, input_dim=8)
    assert len(built.blocks) == 5
    assert built.input_projection.out_features == 32


def test_trainable_parameter_count_skips_frozen_parameters() -> None:
    script = _load_script()
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[0].requires_grad_(False)
    assert script.trainable_parameter_count(model) == 4


def test_an_existing_run_directory_is_refused(tmp_path: Path) -> None:
    script = _load_script()
    with pytest.raises(ValueError, match="already exists"):
        script._require_new_run_directory(tmp_path)
    script._require_new_run_directory(tmp_path / "new")


def test_the_run_directory_records_the_config(tmp_path: Path) -> None:
    script = _load_script()
    run_directory = tmp_path / "runs" / "a"
    document = {"seed": 1, "phases": [{"name": "frozen"}]}

    script._create_run_directory(run_directory, document)

    written = (run_directory / "run_config.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(written) == document


def test_cpu_is_always_an_available_device() -> None:
    _load_script()._require_device(torch.device("cpu"))


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without a GPU")
def test_cuda_without_a_gpu_is_refused() -> None:
    with pytest.raises(RuntimeError, match="no GPU"):
        _load_script()._require_device(torch.device("cuda"))


def test_every_card_costs_one() -> None:
    assert _load_script().card_cost(object()) == 1
