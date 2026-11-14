"""Write this experiment's launch configs into ../configs/.

Two arms that differ only in where the single-card contrastive pairs
come from: R (real decks, `contrastive.<game>`) and T (teacher
neighbours, `teacher_contrastive.<game>`). Each arm scores the other
arm's dojos as held-out dojos (both are head-free, so their TEST loss is
meaningful without training them).

Usage (project root):
    python experiments/2026-11-11_teacher_vs_real_pairs/analysis/make_configs.py pilot
    python experiments/2026-11-11_teacher_vs_real_pairs/analysis/make_configs.py main \
        --lr-r <encoder_lr> --lr-t <encoder_lr> --max-rounds <n>

`pilot` writes the six LR-pilot configs; `main` writes the two main-run
configs once the pilot has picked each arm's LR and measured step times.
"""

import argparse
from pathlib import Path

import yaml

EXPERIMENT = "2026-11-11_teacher_vs_real_pairs"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# Teacher TRAIN anchors per game (confident cards, holdout seed 0, 8/1/1,
# Gwent held out; plans/teacher_full_test_readiness.md step 2.5). The diet
# weight is round(sqrt(anchors)), identical in both arms.
ANCHORS = {
    "mtg": 25_518,
    "yugioh": 11_340,
    "pokemon": 6_033,
    "hearthstone": 4_335,
    "flesh_and_blood": 3_825,
    "digimon": 2_880,
    "one_piece": 1_594,
    "slay_the_spire_2": 451,
    "star_wars_unlimited": 391,
    "dragon_ball_masters": 370,
    "gundam": 362,
    "dragon_ball_fusion_world": 329,
    "lorcana": 265,
    "battle_spirits_saga": 243,
    "dominion": 144,
}
GAMES = list(ANCHORS)
ARM_PREFIX = {"R": "contrastive", "T": "teacher_contrastive"}
OTHER = {"R": "T", "T": "R"}
WHITENING_STATS = f"data/runs/{EXPERIMENT}/whitening_stats.pt"
CACHE_PATH = f"data/feature_cache/{EXPERIMENT}.sqlite"


def diet_table(arm: str) -> dict:
    """A per-game table diet over the arm's own dojos."""
    prefix = ARM_PREFIX[arm]
    rows = [
        {"weight": round(anchors**0.5), "dojo": f"{prefix}.{game}"}
        for game, anchors in ANCHORS.items()
    ]
    return {"rule": "table", "table": rows}


def run_config(
    name: str, arm: str, encoder_lr: float, steps_per_round: int, max_rounds: int, warmup: int,
    keep_round_encoders: bool = False,
) -> dict:
    """One run's config: the shared model, data and budget, the arm's dojos.
    keep_round_encoders: keep every round's encoder (main runs, for the
    matched-step speed comparison); the pilot keeps only the best."""
    return {
        **({"keep_round_encoders": True} if keep_round_encoders else {}),
        "run_directory": f"data/runs/{EXPERIMENT}/{name}",
        "device": "cuda",
        "seed": 0,
        "model": {
            "embedding_head": "residual_mlp",
            "checkpoint": "answerdotai/ModernBERT-base",
            "card_embedding_size": 256,
            "mlp_hidden_dim": 512,
            "mlp_num_blocks": 3,
            "pooling": {"kind": "mean"},
            "enrichment": {"kind": "whitening", "stats_path": WHITENING_STATS},
            "feature_cache": {
                "storage_dtype": "fp16",
                "vram": {"capacity_mib": 3072, "policy": "lfu"},
                "ram": {"capacity_mib": 4096, "policy": "lru", "admission": "first_offer"},
                "disk": {"path": CACHE_PATH, "capacity_mib": 20480, "write": "warm_only"},
            },
        },
        "dojos": [f"{ARM_PREFIX[arm]}.{game}" for game in GAMES],
        "held_out_dojos": [f"{ARM_PREFIX[OTHER[arm]]}.{game}" for game in GAMES],
        "holdout": {"seed": 0, "tier_ratios": [8, 1, 1], "held_out_games": ["gwent"]},
        "teacher": {
            "games": GAMES,
            "method": {"kind": "ppmi_svd", "dim": 128},
            "neighbours": {"kind": "top_k", "k": 32},
            "cooccurrence": {"deck_size_rule": "inverse_size"},
        },
        "regression_loss": "huber",
        "hardware": {
            "max_batch_cost": 512,
            "precision": "fp16",
            "prefetch_depth": 4,
            "torch_cpu_threads": 1,
        },
        "eval_examples_per_dojo": 128,
        "default_augmentation": {"variants": 4},
        "warm_feature_cache": GAMES,
        "phases": [
            {
                "name": "train",
                "diet": diet_table(arm),
                "encoder_trainable": True,
                "encoder_lr": encoder_lr,
                "head_lr": 1.0e-3,
                "lr_schedule": {"warmup_steps": warmup, "decay": "cosine", "min_lr_fraction": 0.1},
                "steps_per_round": steps_per_round,
                "max_rounds": max_rounds,
                "saturation": {
                    "epsilon": 0.001,
                    "patience_rounds": 1000,
                    "reactivation_delta": 0.05,
                    "target_saturated_fraction": 1.0,
                },
            }
        ],
    }


def write(name: str, config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pilot", "main"])
    parser.add_argument("--lr-r", type=float)
    parser.add_argument("--lr-t", type=float)
    parser.add_argument("--max-rounds", type=int)
    args = parser.parse_args()
    if args.stage == "pilot":
        for arm in ("R", "T"):
            for lr in (1.0e-4, 3.0e-4, 1.0e-3):
                name = f"pilot_{arm}_lr{lr:.0e}"
                write(name, run_config(name, arm, lr, steps_per_round=1000, max_rounds=3, warmup=500))
        return
    if args.lr_r is None or args.lr_t is None or args.max_rounds is None:
        parser.error("main needs --lr-r, --lr-t and --max-rounds")
    for arm, lr in (("R", args.lr_r), ("T", args.lr_t)):
        name = f"main_{arm}"
        write(name, run_config(name, arm, lr, steps_per_round=10_000, max_rounds=args.max_rounds, warmup=2000,
                                     keep_round_encoders=True))


if __name__ == "__main__":
    main()
