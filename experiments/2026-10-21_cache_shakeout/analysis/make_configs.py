"""Write this experiment's launch configs from model_by_diet_r2's recorded
single_mixed config: same model, dojos, diet, holdout, seed and batch
budget, with a shorter budget and the run-system settings each run tests.

Usage (from the project root):
    python experiments/2026-10-21_cache_shakeout/analysis/make_configs.py
"""

import copy
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = Path("experiments/2026-10-21_cache_shakeout")
RUNS = Path("data/runs/2026-10-21_cache_shakeout")
SOURCE = Path("experiments/2026-10-11_model_by_diet_r2/configs/single_mixed.yaml")
WHITENING = "data/whitening/modernbert_tokens_holdout_seed0.pt"
GAMES = ["mtg", "pokemon", "hearthstone", "flesh_and_blood", "dominion",
         "slay_the_spire_2", "gwent"]
FEATURE_CACHE = {
    "storage_dtype": "fp16",
    "vram": {"capacity_mib": 2048, "policy": "lru"},
    "ram": {"capacity_mib": 4096, "admission": "first_offer"},
    "disk": {"path": "data/feature_cache/shakeout_2026-10-21.sqlite",
             "capacity_mib": 20480, "write": "warm_only"},
}


def base(name: str, steps_per_round: int, max_rounds: int) -> dict[str, Any]:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["run_directory"] = str(RUNS / name).replace("\\", "/")
    config["model"]["enrichment"]["stats_path"] = WHITENING
    config["hardware"]["prefetch_depth"] = 2
    phase = config["phases"][0]
    phase["steps_per_round"] = steps_per_round
    phase["max_rounds"] = max_rounds
    return config


def cached(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["model"]["feature_cache"] = copy.deepcopy(FEATURE_CACHE)
    config["warm_feature_cache"] = list(GAMES)
    return config


def without_mods(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["mods"] = {name: [] for name in config["dojos"]}
    return config


def write(name: str, config: dict[str, Any]) -> None:
    path = EXPERIMENT / "configs" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(path)


def main() -> None:
    write("pilot", cached(base("pilot", 100, 2)))
    write("cached", cached(base("cached", 1000, 2)))
    write("cached_nocopy", cached(base("cached_nocopy", 1000, 2)))
    write("cached_nomods", without_mods(cached(base("cached_nomods", 1000, 2))))
    write("uncached", base("uncached", 1000, 2))


if __name__ == "__main__":
    main()
