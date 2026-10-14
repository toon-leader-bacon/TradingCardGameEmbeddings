"""Fresh-head (extrinsic) evaluation on Gwent, the game no cell trained on.

Every Gwent dojo is left out of every training diet: the runs' holdout lists
Gwent in held_out_games, which makes every Gwent card VALIDATION-tier. That
holdout leaves a Gwent dojo no TRAIN examples for a fresh head, so here the
Gwent dojos are built with the same seed and ratios (8, 1, 1) but no
held-out game. Gwent cards then split 80/10/10 for the fresh heads, while
the encoders still never saw any Gwent card.

Dojos: every Gwent catalog dojo except contrastive_card_in_contexts.gwent,
which is kept out of every diet and evaluation (src/evaluation/TODO.md); a
dojo with no trainable head or no TRAIN examples is reported and dropped.

Encoders: the six cells and the six untrained controls (seeds 0-2 per
shape). Settings, cache and output format are extrinsic_held_out.py's
(SPEC: 6 rounds x 500 steps, uniform diet, head_lr 1e-3, seed 0).

Writes data/runs/2026-10-11_model_by_diet_r2/extrinsic_gwent/extrinsic/<label>/
(rounds.csv, validation.json), results/extrinsic_gwent_validation.csv (one
row per encoder and dojo) and learning curves into
data/evaluations/2026-10-11_model_by_diet_r2/extrinsic_gwent/plots/.

Run (from the project root, ROCm venv, GPU):
    PYTHONPATH=. python experiments/2026-10-11_model_by_diet_r2/analysis/extrinsic_gwent.py
"""

import csv
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import extrinsic_held_out as base  # noqa: E402

from src.dojos.loss.regression_objective import RegressionObjective  # noqa: E402
from src.evaluation.extrinsic.learning_curves import plot_learning_curves  # noqa: E402
from src.evaluation.extrinsic.run_extrinsic import run_extrinsic  # noqa: E402
from src.schema.splits import Split  # noqa: E402
from src.schema.holdout import HoldoutSpec  # noqa: E402
from src.training.dojo_catalog import DOJO_CATALOG, CardShelf, DojoBuildContext, build_dojos  # noqa: E402
from src.training.plan import HardwareLimits  # noqa: E402

OUT = base.RUNS / "extrinsic_gwent"
PLOTS = Path("data/evaluations/2026-10-11_model_by_diet_r2/extrinsic_gwent/plots")
EXCLUDED = {"contrastive_card_in_contexts.gwent"}
CONTROL_SEEDS = (0, 1, 2)


def gwent_dojos(config) -> list:
    held = config.plan.holdout
    holdout = HoldoutSpec(seed=held.seed, tier_ratios=held.tier_ratios, held_out_games=frozenset())
    context = DojoBuildContext(
        shelf=CardShelf(),
        holdout=holdout,
        card_embedding_size=config.model.card_embedding_size,
        rng_seed=config.plan.seed,
        mod_overrides={},
        staple_thresholds=config.staple_thresholds,
        regression_objective=RegressionObjective.for_kind(config.regression_loss),
    )
    names = sorted(n for n in DOJO_CATALOG if "gwent" in n and n not in EXCLUDED)
    kept = []
    for name in names:
        try:
            (dojo,) = build_dojos([name], context)
        except Exception as error:  # report every dojo that cannot be built, keep the rest
            print(f"drop {name}: build failed: {error}", flush=True)
            continue
        counts = {split.name: dojo.example_count(split) for split in Split}
        has_head = next(iter(dojo.trainable_parameters()), None) is not None
        print(f"{name}: examples {counts} trainable_head={has_head}", flush=True)
        if has_head and counts.get("TRAIN", 0) > 0:
            kept.append(dojo)
        else:
            print(f"drop {name}", flush=True)
    return kept


def encoders() -> list[tuple[str, str, int | None]]:
    rows: list[tuple[str, str, int | None]] = [(cell, cell, None) for cell in base.CELLS]
    for shape in ("single", "multi"):
        rows += [(f"control_{shape}_s{s}", f"{shape}_mixed", s) for s in CONTROL_SEEDS]
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = base.load_config("single_mixed")
    dojos = gwent_dojos(config)
    print("dojos:", [d.name for d in dojos], flush=True)
    if "--list" in sys.argv:
        return
    hardware = HardwareLimits(max_batch_cost=256, precision="fp16")
    cache = base.FrozenFeatureCache()
    rows = []
    for label, cell, seed in encoders():
        run_dir = OUT / "extrinsic" / label
        if not (run_dir / "validation.json").exists():
            if seed is not None:
                torch.manual_seed(seed)
            model = base.run_training.build_model(base.load_config(cell).model).to(base.DEVICE)
            if seed is None:
                state = torch.load(base.RUNS / cell / "latest" / "encoder.pt", map_location=base.DEVICE)
                model.load_state_dict(state)
            base.use_cache(model, cache)
            result = run_extrinsic(
                model, label, dojos, base.SPEC, hardware, OUT, cost_of=base.run_training.card_cost
            )
            print(f"{label}: stopped_early={result.stopped_early_reason} "
                  f"quarantined={sorted(result.quarantined)} cache={len(cache)}", flush=True)
            del model
            torch.cuda.empty_cache()
        payload = json.loads((run_dir / "validation.json").read_text())
        rows.extend(base.validation_rows(label, payload))
    base.RESULTS.mkdir(parents=True, exist_ok=True)
    with (base.RESULTS / "extrinsic_gwent_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", base.RESULTS / "extrinsic_gwent_validation.csv")
    curves = {label: OUT / "extrinsic" / label / "rounds.csv"
              for label in ("control_single_s0", "single_contrastive", "single_metric", "single_mixed",
                            "control_multi_s0", "multi_contrastive", "multi_metric", "multi_mixed")}
    plot_learning_curves(curves, PLOTS)


if __name__ == "__main__":
    main()
