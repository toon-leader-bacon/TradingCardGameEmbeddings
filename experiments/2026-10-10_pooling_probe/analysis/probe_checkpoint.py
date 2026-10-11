"""Score trained runs with probe.py's metrics, on the same cards and decks.

For each run directory: rebuild the model from its recorded
run_config.yaml (scripts/run_training.build_model), load latest/encoder.pt,
and score two views of every card:
- "pipeline": CardEmbeddingPipeline output (text encoder -> enrichment ->
  pooling -> head), before the multi-card attention stack;
- "isolated": model.isolated_embeddings, each card alone through the whole
  model (what evaluation stores).
Rows are appended to results_checkpoints.csv (one per run and view).

Run: PYTHONPATH=. python experiments/2026-10-10_pooling_probe/analysis/probe_checkpoint.py data/runs/<pilot>/<arm> [...]
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import probe  # noqa: E402

from src.training.dojo_catalog import DOJO_CATALOG  # noqa: E402
from src.training.run_config import parse_run_config, read_config_document  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_training", "scripts/run_training.py")
assert _spec is not None and _spec.loader is not None
run_training = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_training)

OUT_CSV = probe.OUT / "results_checkpoints.csv"


def embed_views(run: Path, cards: list) -> dict[str, torch.Tensor]:
    config = parse_run_config(
        read_config_document(run / "run_config.yaml"), tuple(DOJO_CATALOG)
    )
    model = run_training.build_model(config.model).to(probe.DEVICE)
    state = torch.load(run / "latest" / "encoder.pt", map_location=probe.DEVICE)
    model.load_state_dict(state)
    model.eval()
    result: dict[str, torch.Tensor] = {}
    with torch.no_grad(), torch.autocast(probe.DEVICE, dtype=torch.float16):
        pieces = [
            model.card_pipeline.embed(cards[i : i + 64]).float().cpu()
            for i in range(0, len(cards), 64)
        ]
    result["pipeline"] = torch.cat(pieces)
    result["isolated"] = model.isolated_embeddings(cards, "fp16")
    return result


def score(name: str, view: str, x: torch.Tensor, pool: probe.CardPool, labels: list) -> dict:
    mean_cos, top_share = probe.anisotropy(x)
    knn = {n: probe.knn_balanced_accuracy(x[idx], y) for n, kind, idx, y in labels if kind == "cls"}
    linear = {n: probe.linear_probe(x[idx].numpy(), kind, y) for n, kind, idx, y in labels if n != "game"}
    kinds = {n: kind for n, kind, _, _ in labels}
    aucs = probe.deck_auc(x, pool)
    centered_aucs = probe.deck_auc(x - x.mean(0), pool)
    return {
        "run": name,
        "view": view,
        "mean_cos": round(mean_cos, 3),
        "top_pc_share": round(top_share, 3),
        "linear_reg_r2": round(float(np.mean([v for k, v in linear.items() if kinds[k] == "reg"])), 3),
        "linear_cls_bacc": round(float(np.mean([v for k, v in linear.items() if kinds[k] == "cls"])), 3),
        "knn_game_bacc": round(knn["game"], 3),
        "knn_field_bacc": round(float(np.mean([v for k, v in knn.items() if k != "game"])), 3),
        "deck_auc": round(float(np.mean(list(aucs.values()))), 3),
        "deck_auc_centered": round(float(np.mean(list(centered_aucs.values()))), 3),
        **{f"auc.{k}": round(v, 3) for k, v in aucs.items()},
    }


def main() -> None:
    torch.set_num_threads(6)
    pool = probe.build_pool()
    labels = probe.label_sets(pool)
    write_header = not OUT_CSV.exists()
    for run in map(Path, sys.argv[1:]):
        views = embed_views(run, pool.cards)
        for view, x in views.items():
            row = score(run.name, view, x, pool, labels)
            print(
                f"{run.name:20s} {view:8s} | cos {row['mean_cos']:.3f} | lin r2 {row['linear_reg_r2']:.3f} "
                f"cls {row['linear_cls_bacc']:.3f} | knn field {row['knn_field_bacc']:.3f} | "
                f"deck auc {row['deck_auc']:.3f} (centered {row['deck_auc_centered']:.3f})",
                flush=True,
            )
            with open(OUT_CSV, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row), extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)


if __name__ == "__main__":
    main()
