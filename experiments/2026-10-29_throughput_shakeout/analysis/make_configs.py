"""Write this experiment's launch configs into configs/.

Usage (from project root):
    python experiments/2026-10-29_throughput_shakeout/analysis/make_configs.py

The base is 2026-10-24_scale_profile's mtg_n0 launch config (same model,
cache tiers, holdout and optimizer). Each arm changes only the keys named
in ARMS. Whitening stats are the scale_profile ones: fitted on every
non-held-out game's TRAIN-tier cards under the same card tiers.
"""

import copy
from pathlib import Path
from typing import Any

import yaml

_EXPERIMENT = "2026-10-29_throughput_shakeout"
_RUNS = Path("data/runs") / _EXPERIMENT
_CONFIGS = Path("experiments") / _EXPERIMENT / "configs"
_WHITENING = "data/runs/2026-10-24_scale_profile/mtg_n0/whitening_stats.pt"
_THREE_GAMES = ["mtg", "flesh_and_blood", "dominion"]

_BASE: dict[str, Any] = {
    "device": "cuda",
    "seed": 0,
    "model": {
        "embedding_head": "residual_mlp",
        "checkpoint": "answerdotai/ModernBERT-base",
        "card_embedding_size": 256,
        "mlp_hidden_dim": 512,
        "mlp_num_blocks": 3,
        "group_attention": {
            "num_layers": 2,
            "num_heads": 4,
            "ffn_dim": 1024,
            "dropout": 0.1,
            "norm_first": True,
        },
        "pooling": {"kind": "mean"},
        "enrichment": {"kind": "whitening", "stats_path": _WHITENING},
        "feature_cache": {
            "storage_dtype": "fp16",
            "vram": {"capacity_mib": 3072, "policy": "lfu"},
            "ram": {"capacity_mib": 4096, "policy": "lru", "admission": "first_offer"},
            "disk": {"path": "", "capacity_mib": 20480, "write": "warm_only"},
        },
    },
    "dojos": ["contrastive.mtg"],
    "held_out_dojos": [],
    "holdout": {"seed": 0, "tier_ratios": [8, 1, 1], "held_out_games": ["gwent"]},
    "regression_loss": "huber",
    "hardware": {"max_batch_cost": 512, "precision": "fp16", "prefetch_depth": 4},
    "eval_examples_per_dojo": 128,
    "phases": [
        {
            "name": "train",
            "diet": {"rule": "uniform"},
            "encoder_trainable": True,
            "encoder_lr": 0.0001,
            "head_lr": 0.001,
            "steps_per_round": 6000,
            "max_rounds": 1,
            "saturation": {
                "epsilon": 0.001,
                "patience_rounds": 1000,
                "reactivation_delta": 0.05,
                "target_saturated_fraction": 1.0,
            },
        }
    ],
    "default_augmentation": {"probability": 0.1, "variants": 0},
    "warm_feature_cache": ["mtg"],
}

# Three games, keys shuffled once into the binder; dojo mods on top per arm
_THREE: dict[str, Any] = {
    "dojos": [f"contrastive.{game}" for game in _THREE_GAMES],
    "warm_feature_cache": _THREE_GAMES,
    "binder_mods": {
        "pipeline": [{"kind": "shuffle_keys", "probability": 1.0}],
        "seed": 0,
        "save_directory": "data/cache/binder_mods/throughput_shakeout_shuffle_keys",
    },
    "default_augmentation": {"variants": 0},
    "model.feature_cache.disk.path": "data/feature_cache/throughput_shakeout_three.sqlite",
    "phases.0.steps_per_round": 2000,
    "phases.0.max_rounds": 4,
}

# arm -> dotted overrides on _BASE (a "." key path; lists by index)
ARMS: dict[str, dict[str, Any]] = {
    "control_mtg_n0": {
        "model.feature_cache.disk.path": "data/feature_cache/throughput_shakeout_control.sqlite",
    },
    "three_baked_n0": dict(_THREE),
    "three_baked_n4": {**_THREE, "default_augmentation": {"variants": 4}},
    "three_baked_n0_b2048": {
        **_THREE,
        "hardware.max_batch_cost": 2048,
        "phases.0.steps_per_round": 500,
    },
    # Same config as three_baked_n0; OMP_NUM_THREADS=2 comes from env/<arm>.env
    "three_baked_n0_omp2": {**_THREE, "phases.0.max_rounds": 2},
    "three_baked_n0_omp1": {**_THREE, "phases.0.max_rounds": 2},
    # three_baked_n0_omp1 under py-spy (analysis/run_pyspy.sh): every thread
    # with idle stacks, and GIL holders only
    "pyspy_all": {**_THREE, "phases.0.max_rounds": 2},
    "pyspy_gil": {**_THREE, "phases.0.max_rounds": 2},
    "three_baked_n0_omp2_b2048": {
        **_THREE,
        "hardware.max_batch_cost": 2048,
        "phases.0.steps_per_round": 500,
        "phases.0.max_rounds": 2,
    },
}


def _set(document: dict[str, Any], dotted: str, value: Any) -> None:
    """Set document at a dotted path ("a.0.b"), in place."""
    *parents, last = dotted.split(".")
    node: Any = document
    for step in parents:
        node = node[int(step)] if isinstance(node, list) else node[step]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def arm_config(arm: str) -> dict[str, Any]:
    """The launch config for one arm."""
    document = copy.deepcopy(_BASE)
    document["run_directory"] = (_RUNS / arm).as_posix()
    for dotted, value in ARMS[arm].items():
        _set(document, dotted, copy.deepcopy(value))
    return {"run_directory": document.pop("run_directory"), **document}


def main() -> None:
    _CONFIGS.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        path = _CONFIGS / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(arm_config(arm), sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
