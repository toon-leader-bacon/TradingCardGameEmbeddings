"""Zero-shot contrastive scores: every contrastive dojo's VALIDATION loss on
each frozen encoder, with no training.

Contrastive dojos have no head: their loss is computed straight from the
card embeddings. So they cannot take a fresh head (extrinsic_gwent.py drops
them), but they score any frozen encoder directly. This puts the games the
contrastive cells trained on (Flesh and Blood, Slay the Spire 2, MTG,
Pokemon, Dominion) and the games they did not (Gwent, and Hearthstone if the
catalog has its contrastive dojos) on one scale: normalized loss, 1.0 =
chance.

Dojos: every catalog contrastive dojo except contrastive_card_in_contexts.*
(kept out of every diet and evaluation; see src/evaluation/TODO.md), built
with the runs' own holdout and seed. Split: VALIDATION (decks no training
round scored), at most MAX_EXAMPLES per dojo, scored with
src/training/round_evaluation.evaluate_split_losses (the trainer's own
TEST/VALIDATION scorer).

Encoders: the six cells and the six untrained controls (seeds 0-2 per
shape), with the shared frozen-feature cache from extrinsic_held_out.py.

Writes results/zero_shot_contrastive.csv: one row per encoder and dojo
(style, game, whether the game was trained, raw and normalized loss).

Run (from the project root, ROCm venv, GPU):
    PYTHONPATH=. python experiments/2026-10-11_model_by_diet_r2/analysis/zero_shot_contrastive.py
"""

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import extrinsic_held_out as base  # noqa: E402

from src.dojos.dojo import BatchBudget  # noqa: E402
from src.dojos.loss.regression_objective import RegressionObjective  # noqa: E402
from src.schema.splits import Split  # noqa: E402
from src.training.dojo_catalog import DOJO_CATALOG, build_dojos  # noqa: E402
from src.training.dojo_recipes.card_shelf import CardShelf  # noqa: E402
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext  # noqa: E402
from src.training.round_evaluation import evaluate_split_losses  # noqa: E402

MAX_EXAMPLES = 1024
CONTROL_SEEDS = (0, 1, 2)
TRAINED_GAMES = {"flesh_and_blood", "slay_the_spire_2", "mtg", "pokemon", "dominion"}


def contrastive_dojos(config) -> list:
    context = DojoBuildContext(
        shelf=CardShelf(),
        holdout=config.plan.holdout,
        card_embedding_size=config.model.card_embedding_size,
        rng_seed=config.plan.seed,
        mod_overrides={},
        staple_thresholds=config.staple_thresholds,
        regression_objective=RegressionObjective.for_kind(config.regression_loss),
    )
    names = sorted(n for n in DOJO_CATALOG
                   if n.startswith("contrastive") and not n.startswith("contrastive_card_in_contexts"))
    return build_dojos(names, context)


def encoders() -> list[tuple[str, str, int | None]]:
    rows: list[tuple[str, str, int | None]] = [(cell, cell, None) for cell in base.CELLS]
    for shape in ("single", "multi"):
        rows += [(f"control_{shape}_s{s}", f"{shape}_mixed", s) for s in CONTROL_SEEDS]
    return rows


def main() -> None:
    config = base.load_config("single_mixed")
    dojos = contrastive_dojos(config)
    print("dojos:", [d.name for d in dojos], flush=True)
    budget = BatchBudget(256, base.run_training.card_cost)
    cache = base.FrozenFeatureCache()
    rows = []
    for label, cell, seed in encoders():
        if seed is not None:
            torch.manual_seed(seed)
        model = base.run_training.build_model(base.load_config(cell).model).to(base.DEVICE)
        if seed is None:
            state = torch.load(base.RUNS / cell / "latest" / "encoder.pt", map_location=base.DEVICE)
            model.load_state_dict(state)
        base.use_cache(model, cache)
        model.eval()
        with torch.no_grad():
            losses = evaluate_split_losses(model, dojos, split=Split.VALIDATION, budget=budget,
                                           max_examples=MAX_EXAMPLES, precision="fp16")
        for name, split_loss in sorted(losses.items()):
            style, game = name.split(".")
            rows.append({
                "encoder": label,
                "dojo": name,
                "style": style,
                "game": game,
                "game_trained": game in TRAINED_GAMES,
                "loss": round(split_loss.loss, 4),
                "normalized": round(split_loss.normalized, 4),
            })
        print(label, {n: round(s.normalized, 3) for n, s in losses.items() if n.endswith("gwent")},
              f"cache={len(cache)}", flush=True)
        del model
        torch.cuda.empty_cache()
    base.RESULTS.mkdir(parents=True, exist_ok=True)
    with (base.RESULTS / "zero_shot_contrastive.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", base.RESULTS / "zero_shot_contrastive.csv")


if __name__ == "__main__":
    main()
