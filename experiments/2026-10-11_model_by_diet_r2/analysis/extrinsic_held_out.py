"""Fresh-head (extrinsic) evaluation of every cell on the 13 held-out dojos.

During training the held-out dojos were scored every round but their heads
were never trained, so their TEST loss is that of a randomly initialized
head and says nothing about transfer. Here each encoder is frozen and the
held-out dojos' heads are trained from identical starting weights with
src/evaluation/extrinsic/run_extrinsic.py, then each head (restored to its
best TEST round) is scored on VALIDATION.

Encoders: the six cells' latest/encoder.pt, plus one untrained control per
model shape (build_model under torch.manual_seed(0), as in probe_cells.py).

Speed: the frozen part of every model (ModernBERT, token whitening, mean
pooling) is identical across encoders, so its output is cached per card
the first time any encoder sees the card; each step then runs only the
head, the group attention (multi-card) and the dojo head.

Writes data/runs/2026-10-11_model_by_diet_r2/extrinsic/extrinsic/<label>/
(rounds.csv, validation.json) and results/extrinsic_validation.csv: one row
per (encoder, dojo) with the VALIDATION loss normalized by the dojo's
baseline (1.0 = mean predictor / prior), and the best TEST round.

Run (from the project root, ROCm venv, GPU):
    PYTHONPATH=. python experiments/2026-10-11_model_by_diet_r2/analysis/extrinsic_held_out.py
"""

import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import MethodType

import torch

from src.dojos.loss.regression_objective import RegressionObjective
from src.encoder_model.card_serialization import serialize_card_to_json
from src.evaluation.extrinsic.run_extrinsic import ExtrinsicSpec, run_extrinsic
from src.training.dojo_catalog import DOJO_CATALOG, CardShelf, DojoBuildContext, build_dojos
from src.training.plan import HardwareLimits, SaturationSpec, Uniform
from src.training.run_config import parse_run_config, read_config_document

_spec = importlib.util.spec_from_file_location("run_training", "scripts/run_training.py")
assert _spec is not None and _spec.loader is not None
run_training = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_training)

RUNS = Path("data/runs/2026-10-11_model_by_diet_r2")
OUT = RUNS / "extrinsic"
RESULTS = Path("experiments/2026-10-11_model_by_diet_r2/results")
CELLS = (
    "single_contrastive",
    "single_metric",
    "single_mixed",
    "multi_contrastive",
    "multi_metric",
    "multi_mixed",
)
SPEC = ExtrinsicSpec(
    diet_rule=Uniform(),
    head_lr=1e-3,
    steps_per_round=500,
    max_rounds=6,
    saturation=SaturationSpec(
        epsilon=0.001, patience_rounds=1000, reactivation_delta=1.0, target_saturated_fraction=1.0
    ),
    eval_examples_per_dojo=512,
    seed=0,
)
DEVICE = "cuda"
BATCH = 64


class FrozenFeatureCache:
    """Whitened mean-pooled token states per card, shared by every encoder."""

    def __init__(self) -> None:
        self._rows: dict = {}

    def features(self, pipeline, cards) -> torch.Tensor:
        """(len(cards), 768) on DEVICE; computes and stores missing cards."""
        missing = list({c.nocab_uuid: c for c in cards if c.nocab_uuid not in self._rows}.values())
        if missing:
            with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16):
                for start in range(0, len(missing), BATCH):
                    chunk = missing[start : start + BATCH]
                    encoding = pipeline.text_encoder.encode([serialize_card_to_json(c) for c in chunk])
                    if pipeline.enrichment is not None:
                        encoding = pipeline.enrichment(encoding)
                    pooled = pipeline.pooling(encoding).float()
                    for card, row in zip(chunk, pooled):
                        self._rows[card.nocab_uuid] = row
        return torch.stack([self._rows[c.nocab_uuid] for c in cards])

    def __len__(self) -> int:
        return len(self._rows)


def use_cache(model, cache: FrozenFeatureCache) -> None:
    def embed(self, cards):
        return self.embedding_head(cache.features(self, cards))

    model.card_pipeline.embed = MethodType(embed, model.card_pipeline)


def load_config(cell: str):
    return parse_run_config(
        read_config_document(RUNS / cell / "run_config.yaml"), tuple(DOJO_CATALOG)
    )


def encoders() -> list[tuple[str, str, int | None]]:
    """(label, config cell, control seed or None for the trained weights)."""
    rows: list[tuple[str, str, int | None]] = [(cell, cell, None) for cell in CELLS]
    rows += [("control_single_s0", "single_mixed", 0), ("control_multi_s0", "multi_mixed", 0)]
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config("single_mixed")
    context = DojoBuildContext(
        shelf=CardShelf(),
        holdout=config.plan.holdout,
        card_embedding_size=config.model.card_embedding_size,
        rng_seed=config.plan.seed,
        mod_overrides=config.mod_overrides,
        staple_thresholds=config.staple_thresholds,
        regression_objective=RegressionObjective.for_kind(config.regression_loss),
    )
    names = sorted(config.plan.held_out_dojos)
    dojos = build_dojos(names, context)
    hardware = HardwareLimits(max_batch_cost=256, precision="fp16")
    cache = FrozenFeatureCache()
    rows = []
    for label, cell, seed in encoders():
        run_dir = OUT / "extrinsic" / label
        if not (run_dir / "validation.json").exists():
            if seed is not None:
                torch.manual_seed(seed)
            model = run_training.build_model(load_config(cell).model).to(DEVICE)
            if seed is None:
                state = torch.load(RUNS / cell / "latest" / "encoder.pt", map_location=DEVICE)
                model.load_state_dict(state)
            use_cache(model, cache)
            result = run_extrinsic(
                model, label, dojos, SPEC, hardware, OUT, cost_of=run_training.card_cost
            )
            print(f"{label}: stopped_early={result.stopped_early_reason} "
                  f"quarantined={sorted(result.quarantined)} cache={len(cache)}", flush=True)
            del model
            torch.cuda.empty_cache()
        payload = json.loads((run_dir / "validation.json").read_text())
        rows.extend(validation_rows(label, payload))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "extrinsic_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", RESULTS / "extrinsic_validation.csv")


def validation_rows(label: str, payload: dict) -> list[dict]:
    """One row per dojo from a validation.json (format: see run_extrinsic)."""
    quarantined = set(payload["quarantined"])
    return [
        {
            "encoder": label,
            "dojo": dojo,
            "validation_normalized": round(value, 4),
            "validation_loss": round(payload["losses"][dojo], 4),
            "best_test_round": payload["best_test_rounds"].get(dojo, ""),
            "quarantined": dojo in quarantined,
        }
        for dojo, value in sorted(payload["normalized_losses"].items())
    ]


if __name__ == "__main__":
    main()
