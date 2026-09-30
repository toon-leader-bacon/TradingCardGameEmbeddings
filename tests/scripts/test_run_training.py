import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch
import yaml

from src.training.run_config import ModelKind

_SCRIPT = Path("scripts/run_training.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_training_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_model_kind_has_a_constructor() -> None:
    script = _load_script()
    assert set(script._MODEL_CONSTRUCTORS) == set(ModelKind)


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
