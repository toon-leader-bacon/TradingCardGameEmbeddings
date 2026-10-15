import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

from src.encoder_model.enrichment.token_whitening import TokenWhitening
from src.encoder_model.pooling.attention_pooling import AttentionPooling
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.encoder_model.pooling.transformer_pooling import TransformerPooling
from src.encoder_model.whitening.whitening_stats import WhiteningStats
from src.schema.holdout import HoldoutSpec
from src.training.model_spec import (
    AttentionPoolingSpec,
    EmbeddingHeadKind,
    MeanPoolingSpec,
    ModelSpec,
    TokenWhiteningSpec,
    TransformerPoolingSpec,
)
from tests.encoder_model.fakes import HIDDEN, FakeTextEncoder

_SCRIPT = Path("scripts/run_training.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_training_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("head", list(EmbeddingHeadKind))
def test_every_embedding_head_kind_builds_a_card_pipeline(
    head: EmbeddingHeadKind,
) -> None:
    script = _load_script()
    built = script.build_card_pipeline(ModelSpec(head, "ckpt", 16), FakeTextEncoder())
    assert built.output_dim == 16
    assert built.pooling.input_dim == HIDDEN


def test_mean_pooling_feeds_the_linear_and_residual_heads() -> None:
    script = _load_script()
    for head in (EmbeddingHeadKind.LINEAR_PROJECTION, EmbeddingHeadKind.RESIDUAL_MLP):
        built = script.build_card_pipeline(
            ModelSpec(head, "ckpt", 16), FakeTextEncoder()
        )
        assert isinstance(built.pooling, MeanPooling)


@pytest.mark.parametrize(
    ("pooling", "pooling_class", "pooled_width"),
    [
        (MeanPoolingSpec(), MeanPooling, HIDDEN),
        (AttentionPoolingSpec(3), AttentionPooling, 3 * HIDDEN),
        (
            TransformerPoolingSpec(AttentionPoolingSpec(2), num_heads=2),
            TransformerPooling,
            2 * HIDDEN,
        ),
    ],
)
def test_the_pooling_spec_picks_the_pooling_and_sizes_the_head(
    pooling: Any, pooling_class: type, pooled_width: int
) -> None:
    script = _load_script()
    spec = ModelSpec(EmbeddingHeadKind.LINEAR_PROJECTION, "ckpt", 16, pooling=pooling)
    built = script.build_card_pipeline(spec, FakeTextEncoder())
    assert isinstance(built.pooling, pooling_class)
    assert built.embedding_head.input_dim == pooled_width
    assert built.enrichment is None


def _whitening_spec(tmp_path: Path, holdout: HoldoutSpec) -> ModelSpec:
    stats = WhiteningStats(
        checkpoint="ckpt",
        holdout=holdout,
        card_count=4,
        mean=torch.zeros(HIDDEN),
        projection=torch.eye(HIDDEN)[:, :4],
    )
    path = tmp_path / "stats.pt"
    stats.save(path)
    return ModelSpec(
        EmbeddingHeadKind.LINEAR_PROJECTION,
        "ckpt",
        16,
        enrichment=TokenWhiteningSpec(path),
    )


def test_token_whitening_runs_before_the_pooling(tmp_path: Path) -> None:
    script = _load_script()
    spec = _whitening_spec(tmp_path, HoldoutSpec.no_holdout())
    built = script.build_card_pipeline(spec, FakeTextEncoder())
    assert isinstance(built.enrichment, TokenWhitening)
    assert built.pooling.input_dim == 4


def test_stats_from_another_checkpoint_are_refused(tmp_path: Path) -> None:
    script = _load_script()
    spec = replace(_whitening_spec(tmp_path, HoldoutSpec.no_holdout()), checkpoint="x")
    with pytest.raises(ValueError, match="fitted on 'ckpt'"):
        script.build_card_pipeline(spec, FakeTextEncoder())
    with pytest.raises(ValueError, match="fitted on 'ckpt'"):
        script.require_whitening_fits_run(spec, HoldoutSpec.no_holdout())


def test_stats_from_another_holdout_are_refused(tmp_path: Path) -> None:
    script = _load_script()
    spec = _whitening_spec(tmp_path, HoldoutSpec.no_holdout(seed=1))
    script.require_whitening_fits_run(spec, HoldoutSpec.no_holdout(seed=1))
    with pytest.raises(ValueError, match="holdout"):
        script.require_whitening_fits_run(spec, HoldoutSpec.no_holdout(seed=2))


def test_a_run_without_whitening_needs_no_stats() -> None:
    script = _load_script()
    spec = ModelSpec(EmbeddingHeadKind.LINEAR_PROJECTION, "ckpt", 16)
    script.require_whitening_fits_run(spec, HoldoutSpec.no_holdout())


def test_the_residual_mlp_head_takes_its_sizes() -> None:
    script = _load_script()
    spec = ModelSpec(EmbeddingHeadKind.RESIDUAL_MLP, "ckpt", 16, 32, 5)
    built = script.build_card_pipeline(spec, FakeTextEncoder()).embedding_head
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


def test_a_whitening_run_records_and_copies_its_own_stats(tmp_path: Path) -> None:
    script = _load_script()
    spec = _whitening_spec(tmp_path, HoldoutSpec.no_holdout())
    run_directory = tmp_path / "run"
    config = SimpleNamespace(model=spec, run_directory=run_directory)
    document = {
        "model": {"enrichment": {"kind": "whitening", "stats_path": "elsewhere.pt"}},
        "seed": 1,
    }
    recorded = script._with_run_whitening_stats(document, config)
    copy_path = run_directory / "whitening_stats.pt"
    assert recorded["model"]["enrichment"]["stats_path"] == copy_path.as_posix()
    assert recorded["model"]["enrichment"]["kind"] == "whitening"
    assert document["model"]["enrichment"]["stats_path"] == "elsewhere.pt"
    run_directory.mkdir()
    script._copy_whitening_stats(config)
    assert WhiteningStats.load(copy_path).checkpoint == "ckpt"


def test_a_run_without_whitening_records_its_config_unchanged(tmp_path: Path) -> None:
    script = _load_script()
    spec = ModelSpec(EmbeddingHeadKind.LINEAR_PROJECTION, "ckpt", 16)
    config = SimpleNamespace(model=spec, run_directory=tmp_path)
    document = {"model": {"embedding_head": "linear_projection"}}
    assert script._with_run_whitening_stats(document, config) is document
    script._copy_whitening_stats(config)
    assert list(tmp_path.iterdir()) == []
