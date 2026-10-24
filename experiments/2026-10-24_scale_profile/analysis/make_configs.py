"""Write this experiment's launch configs from model_by_diet_r2's recorded
multi_mixed config: the same multi-card model, dojos, diet, holdout and
seed, plus the feature cache, prefetch, batch budget and augmentation
rates this experiment sets. The arms differ only in
default_augmentation.variants (0: augmentation off; 4: bounded).

Usage (from the project root):
    python experiments/2026-10-24_scale_profile/analysis/make_configs.py [name ...]
"""

import copy
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

EXPERIMENT = Path("experiments/2026-10-24_scale_profile")
RUNS = Path("data/runs/2026-10-24_scale_profile")
SOURCE = Path("experiments/2026-10-11_model_by_diet_r2/configs/multi_mixed.yaml")
WHITENING = "data/whitening/modernbert_tokens_holdout_seed0.pt"
GAMES = ["mtg", "pokemon", "hearthstone", "flesh_and_blood", "dominion",
         "slay_the_spire_2", "gwent"]
# Every arm warms the same games into its own new disk file, so every
# startup measures the same cold warm-up
MAX_BATCH_COST = 512
PREFETCH_DEPTH = 4
PROBABILITY = 0.1


def feature_cache(name: str) -> dict[str, Any]:
    """VRAM 3 GiB LFU (about 2.1M rows: the whole corpus and every bounded
    variant fit), pinned RAM 4 GiB LRU, and a new SQLite file per run."""
    return {
        "storage_dtype": "fp16",
        "vram": {"capacity_mib": 3072, "policy": "lfu"},
        "ram": {"capacity_mib": 4096, "policy": "lru", "admission": "first_offer"},
        "disk": {"path": f"data/feature_cache/scale_profile_{name}.sqlite",
                 "capacity_mib": 20480, "write": "warm_only"},
    }


def mixed(name: str, variants: int, steps_per_round: int, max_rounds: int) -> dict[str, Any]:
    """r2's multi_mixed recipe with this experiment's run settings."""
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config["run_directory"] = str(RUNS / name).replace("\\", "/")
    config["model"]["enrichment"]["stats_path"] = WHITENING
    config["model"]["feature_cache"] = feature_cache(name)
    config["hardware"]["max_batch_cost"] = MAX_BATCH_COST
    config["hardware"]["prefetch_depth"] = PREFETCH_DEPTH
    config["default_augmentation"] = {"probability": PROBABILITY, "variants": variants}
    config["warm_feature_cache"] = list(GAMES)
    phase = config["phases"][0]
    phase["steps_per_round"] = steps_per_round
    phase["max_rounds"] = max_rounds
    return config


def mtg_contrastive(name: str, variants: int, steps: int) -> dict[str, Any]:
    """The same model and run settings on contrastive.mtg alone (single-card
    InfoNCE over MTG decks): the rate a full MTG anchor pass would run at."""
    config = mixed(name, variants, steps, 1)
    config["dojos"] = ["contrastive.mtg"]
    config["held_out_dojos"] = []
    config["warm_feature_cache"] = ["mtg"]
    phase = config["phases"][0]
    phase["diet"] = {"rule": "uniform"}
    return config


CONFIGS: dict[str, Callable[[], dict[str, Any]]] = {
    "pilot": lambda: mixed("pilot", 4, 300, 2),
    "mixed_n0": lambda: mixed("mixed_n0", 0, 5500, 6),
    "mixed_n4": lambda: mixed("mixed_n4", 4, 5500, 6),
    "mtg_n0": lambda: mtg_contrastive("mtg_n0", 0, 6000),
    "mtg_n4": lambda: mtg_contrastive("mtg_n4", 4, 6000),
}


def write(name: str, config: dict[str, Any]) -> None:
    path = EXPERIMENT / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(path)


def main() -> None:
    """Writes the named configs (all when none are named)."""
    for name in sys.argv[1:] or CONFIGS:
        write(name, CONFIGS[name]())


if __name__ == "__main__":
    main()
