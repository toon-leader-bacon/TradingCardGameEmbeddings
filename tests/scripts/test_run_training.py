import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

from src.encoder_model.enrichment.token_whitening import TokenWhitening
from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.pooling.attention_pooling import AttentionPooling
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.encoder_model.pooling.transformer_pooling import TransformerPooling
from src.encoder_model.whitening.whitening_stats import WhiteningStats
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.diet.prefetching_batch_streams import NO_BATCH_WARMING
from src.training.plan import HardwareLimits
from src.training.model_spec import (
    AttentionPoolingSpec,
    EmbeddingHeadKind,
    DiskTierSpec,
    FeatureCacheSpec,
    MemoryTierSpec,
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
    with pytest.raises(ValueError, match="card tiers"):
        script.require_whitening_fits_run(spec, HoldoutSpec.no_holdout(seed=2))


def test_stats_under_another_deck_split_are_accepted(tmp_path: Path) -> None:
    # Whitening samples cards, never decks: only card tiers must match
    script = _load_script()
    fitted_under = HoldoutSpec(1, (8, 1, 1), frozenset(), deck_split_ratios=None)
    spec = _whitening_spec(tmp_path, fitted_under)
    run_holdout = HoldoutSpec(1, (8, 1, 1), frozenset(), deck_split_ratios=(6, 2, 2))
    script.require_whitening_fits_run(spec, run_holdout)


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


def _seeded_weights(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, seed: int
) -> list[torch.Tensor]:
    monkeypatch.setattr(script, "PretrainedTextEncoder", lambda **_: FakeTextEncoder())
    spec = ModelSpec(EmbeddingHeadKind.RESIDUAL_MLP, "ckpt", 16)
    model = script.build_model(spec, seed=seed)
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def test_a_seeded_model_starts_from_the_same_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    first = _seeded_weights(script, monkeypatch, seed=3)
    torch.rand(100)  # whatever ran in between does not matter
    second = _seeded_weights(script, monkeypatch, seed=3)
    other = _seeded_weights(script, monkeypatch, seed=4)

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))


def test_a_seeded_build_leaves_the_global_rng_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    _seeded_weights(script, monkeypatch, seed=3)
    assert torch.equal(torch.rand(3), expected)


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
    _load_script().require_device(torch.device("cpu"))


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without a GPU")
def test_cuda_without_a_gpu_is_refused() -> None:
    with pytest.raises(RuntimeError, match="no GPU"):
        _load_script().require_device(torch.device("cuda"))


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


def test_a_feature_cache_spec_attaches_a_sized_cache() -> None:
    script = _load_script()
    cache_spec = FeatureCacheSpec(
        storage_dtype=FeatureStorageDtype.FP16, vram=MemoryTierSpec(1)
    )
    spec = ModelSpec(
        EmbeddingHeadKind.LINEAR_PROJECTION, "ckpt", 16, feature_cache=cache_spec
    )
    cache = script.build_card_pipeline(spec, FakeTextEncoder()).feature_cache
    assert cache is not None
    # 1 MiB of HIDDEN-wide fp16 rows
    assert cache.stats().tiers[0].capacity_rows == 2**20 // (HIDDEN * 2)
    assert cache.fingerprint.pooling == "mean"


def test_no_feature_cache_spec_builds_no_cache() -> None:
    script = _load_script()
    spec = ModelSpec(EmbeddingHeadKind.LINEAR_PROJECTION, "ckpt", 16)
    assert script.build_card_pipeline(spec, FakeTextEncoder()).feature_cache is None


class _WarmRecorder:
    """Fake model: records warm() calls and reports no cache stats."""

    def __init__(self) -> None:
        self.warmed: list[tuple[int, str]] = []

    def warm(self, cards: list[Any], precision: str) -> None:
        self.warmed.append((len(cards), precision))

    def feature_cache_stats(self) -> None:
        return None


class _FakeShelf:
    """Fake CardShelf: every game's binder holds `count` placeholder cards."""

    def __init__(self, count_by_game: dict[GameId, int]) -> None:
        self._count_by_game = count_by_game

    def card_binder(self, game: GameId) -> Any:
        count = self._count_by_game[game]
        return SimpleNamespace(all_cards=lambda source_game: [object()] * count)


def test_warm_feature_cache_warms_every_listed_game_at_the_run_precision() -> None:
    script = _load_script()
    model = _WarmRecorder()
    config = SimpleNamespace(
        warm_feature_cache=(GameId.GWENT, GameId.MTG),
        limits=SimpleNamespace(precision="fp16"),
    )
    shelf = _FakeShelf({GameId.GWENT: 3, GameId.MTG: 5})
    script.warm_feature_cache(model, config, shelf)
    assert model.warmed == [(3, "fp16"), (5, "fp16")]


def test_warm_feature_cache_without_games_does_nothing() -> None:
    script = _load_script()
    model = _WarmRecorder()
    config = SimpleNamespace(
        warm_feature_cache=(), limits=SimpleNamespace(precision="fp16")
    )
    script.warm_feature_cache(model, config, _FakeShelf({}))
    assert model.warmed == []


def test_a_disk_tier_needs_a_text_encoder_with_a_revision(tmp_path: Path) -> None:
    script = _load_script()
    disk = DiskTierSpec(tmp_path / "f.sqlite", 1)
    spec = ModelSpec(
        EmbeddingHeadKind.LINEAR_PROJECTION,
        "ckpt",
        16,
        feature_cache=FeatureCacheSpec(disk=disk),
    )
    # FakeTextEncoder has no source, so its rows may not persist
    with pytest.raises(ValueError, match="known revision"):
        script.build_card_pipeline(spec, FakeTextEncoder())


def test_closing_feature_cache_closes_even_when_the_body_raises() -> None:
    script = _load_script()
    closed: list[bool] = []
    model = SimpleNamespace(close_feature_cache=lambda: closed.append(True))
    with pytest.raises(RuntimeError):
        with script.closing_feature_cache(model):
            raise RuntimeError("boom")
    assert closed == [True]


def _model_with(cache: object) -> Any:
    return SimpleNamespace(card_pipeline=SimpleNamespace(feature_cache=cache))


def test_batch_warmer_for_warms_nothing_without_prefetch_or_without_a_cache() -> None:
    script = _load_script()
    cached = _model_with(SimpleNamespace())
    off = HardwareLimits(max_batch_cost=8, precision="fp16")
    on = HardwareLimits(max_batch_cost=8, precision="fp16", prefetch_depth=4)
    assert script.batch_warmer_for(cached, off) is NO_BATCH_WARMING
    assert script.batch_warmer_for(_model_with(None), on) is NO_BATCH_WARMING


def test_batch_warmer_for_warms_a_cached_model_and_checks_its_precision() -> None:
    script = _load_script()
    fingerprint = SimpleNamespace(compute_precision="fp16")
    model = _model_with(SimpleNamespace(fingerprint=fingerprint))
    warmer = script.batch_warmer_for(
        model, HardwareLimits(max_batch_cost=8, precision="fp16", prefetch_depth=4)
    )
    assert warmer is not NO_BATCH_WARMING
    with pytest.raises(ValueError, match="built for fp16"):
        script.batch_warmer_for(
            model, HardwareLimits(max_batch_cost=8, precision="fp32", prefetch_depth=4)
        )
