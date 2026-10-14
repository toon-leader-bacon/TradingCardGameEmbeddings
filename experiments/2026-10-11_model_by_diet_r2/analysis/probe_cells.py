"""Per-card probe of every trained cell and the untrained controls.

Wraps the pooling probe's metrics
(experiments/2026-10-10_pooling_probe/analysis/probe.py) on the same card
pool and decks, so the full-probe numbers compare with the pilot's, and adds
the Method's split:

- full probe: deck AUC (0.5 = chance) and linear-cls balanced accuracy
  over every probe field, as in the pilot;
- deck AUC by anchor tier (TRAIN against VALIDATION/TEST under the
  experiment's holdout), with a 95% bootstrap interval over decks;
- the out-of-sample slice: Gwent's deck AUC, and linear-cls on
  non-TRAIN-tier cards over the fields no dojo trained (Hearthstone class,
  Flesh and Blood pitch, Slay the Spire 2 type), on every card of those
  three games rather than the small pool.

The frozen part of every model (ModernBERT, token whitening, mean pooling)
has no trainable weights and the same whitening stats in every run, so it
is computed once per card and cached. Each model then runs only its own
embedding head and, for multi-card, its attention over a length-one group
(the "isolated" view, model.isolated_embeddings). Each trained cell
is checked: the cached path against the real one on 64 cards.

Models scored:
- r2_<cell>: each cell's latest/encoder.pt (the round-3 weights);
- r2_control_<model>_s<seed>: build_model from the cell's config under
  torch.manual_seed(seed), untrained. Training never seeds the model's
  initial weights, so no control reproduces a cell's starting point;
  three seeds per shape show how much a random initialization moves the
  probe;
- r2_reference_whitened_mean: the head-free 768-wide whitened mean vector.

Views: "isolated" (the decision view) and "pipeline" (before group
attention; identical to isolated for single-card).

Writes into results/:
    probe_summary.csv       one row per (model, view)
    probe_by_field.csv      one row per (model, view, field): linear and kNN
    probe_deck_by_game.csv  one row per (model, view, game): deck AUC

Run (from the project root, ROCm venv, GPU):
    PYTHONPATH=. python experiments/2026-10-11_model_by_diet_r2/analysis/probe_cells.py
"""

import csv
import importlib.util
import random
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch

PROBE_DIR = Path("experiments/2026-10-10_pooling_probe/analysis")
sys.path.insert(0, str(PROBE_DIR))
import probe  # noqa: E402

from src.data_refinement.card_binder.card_binder import CardBinder  # noqa: E402
from src.encoder_model.card_serialization import serialize_card_to_json  # noqa: E402
from src.schema.game_id import GameId  # noqa: E402
from src.schema.holdout import CardTier  # noqa: E402
from src.training.dojo_catalog import DOJO_CATALOG  # noqa: E402
from src.training.run_config import parse_run_config, read_config_document  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_training", "scripts/run_training.py")
assert _spec is not None and _spec.loader is not None
run_training = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_training)

RUNS = Path("data/runs/2026-10-11_model_by_diet_r2")
RESULTS = Path("experiments/2026-10-11_model_by_diet_r2/results")
CACHE = RUNS / "probe" / "pooled.pt"
CELLS = (
    "single_contrastive",
    "single_metric",
    "single_mixed",
    "multi_contrastive",
    "multi_metric",
    "multi_mixed",
)
CONTROL_SEEDS = (0, 1, 2)
# Probe fields no dojo trains (their dojos are held out), per the Method
HELD_OUT_FIELDS = (
    (GameId.HEARTHSTONE, "class"),
    (GameId.FLESH_AND_BLOOD, "pitch"),
    (GameId.SLAY_THE_SPIRE_2, "type"),
)
BOOTSTRAP = 1000
BATCH = 64


def load_config(cell: str):
    return parse_run_config(
        read_config_document(RUNS / cell / "run_config.yaml"), tuple(DOJO_CATALOG)
    )


def slice_cards() -> list:
    """Every card of the three held-out-field games."""
    cards = []
    for game, _ in HELD_OUT_FIELDS:
        binder = CardBinder.load([CardBinder.default_output_path(game)])
        unknown = CardBinder.unknown_card_uuid(game)
        cards.extend(c for c in binder.all_cards(game) if c.nocab_uuid != unknown)
    return cards


def frozen_features(model, cards: list) -> torch.Tensor:
    """Whitened mean-pooled token states (n, 768): the shared frozen part."""
    pipeline = model.card_pipeline
    pieces = []
    with torch.no_grad(), torch.autocast(probe.DEVICE, dtype=torch.float16):
        for start in range(0, len(cards), BATCH):
            chunk = cards[start : start + BATCH]
            texts = [serialize_card_to_json(c) for c in chunk]
            encoding = pipeline.text_encoder.encode(texts)
            if pipeline.enrichment is not None:
                encoding = pipeline.enrichment(encoding)
            pieces.append(pipeline.pooling(encoding).float().cpu())
    return torch.cat(pieces)


def use_cache(model, pooled: torch.Tensor, row_of: dict) -> None:
    """Make model.card_pipeline.embed read the cached frozen features."""

    def embed(self, cards):
        rows = torch.tensor([row_of[c.nocab_uuid] for c in cards])
        x = pooled[rows].to(probe.DEVICE)
        return self.embedding_head(x)

    model.card_pipeline.embed = MethodType(embed, model.card_pipeline)


def views_of(model, cards: list) -> dict[str, torch.Tensor]:
    """'isolated' (whole model, each card alone) and 'pipeline' (pre-attention)."""
    model.eval()
    with torch.no_grad(), torch.autocast(probe.DEVICE, dtype=torch.float16):
        pipeline = torch.cat(
            [
                model.card_pipeline.embed(cards[i : i + 1024]).float().cpu()
                for i in range(0, len(cards), 1024)
            ]
        )
    isolated = torch.cat(
        [model.isolated_embeddings(cards[i : i + 1024], "fp16") for i in range(0, len(cards), 1024)]
    )
    return {"isolated": isolated, "pipeline": pipeline}


def per_deck_auc(x: torch.Tensor, pool: probe.CardPool, anchor_ok) -> list[tuple[str, float]]:
    """(game, mean AUC over the deck's anchors that pass anchor_ok) per deck."""
    unit = torch.nn.functional.normalize(x, dim=1)
    games = [c.source_game for c in pool.cards]
    out = []
    for game, members in pool.decks:
        member_set = set(members)
        others = [i for i, g in enumerate(games) if g == game and i not in member_set]
        anchors = [a for a in range(len(members)) if anchor_ok(members[a])]
        if not others or not anchors:
            continue
        sims_pos = unit[members] @ unit[members].T
        neg_sorted = (unit[members] @ unit[others].T).sort(dim=1).values
        aucs = []
        for a in anchors:
            positives = torch.cat([sims_pos[a, :a], sims_pos[a, a + 1 :]])
            below = torch.searchsorted(neg_sorted[a].contiguous(), positives.contiguous())
            aucs.append((below.float() / neg_sorted.shape[1]).mean().item())
        out.append((game.value, float(np.mean(aucs))))
    return out


def mean_of_games(rows: list[tuple[str, float]]) -> float:
    by_game: dict[str, list[float]] = {}
    for game, value in rows:
        by_game.setdefault(game, []).append(value)
    return float(np.mean([np.mean(v) for v in by_game.values()]))


def bootstrap_interval(rows: list[tuple[str, float]]) -> tuple[float, float]:
    """95% interval of the mean-of-games deck AUC, resampling decks within game."""
    rng = random.Random(0)
    by_game: dict[str, list[float]] = {}
    for game, value in rows:
        by_game.setdefault(game, []).append(value)
    means = []
    for _ in range(BOOTSTRAP):
        means.append(
            np.mean([np.mean(rng.choices(v, k=len(v))) for v in by_game.values()])
        )
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def slice_labels(cards: list, tiers: list[CardTier]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(field, indices, labels) for each held-out field, non-TRAIN cards only."""
    out = []
    for game, field in HELD_OUT_FIELDS:
        extract = dict((n, f) for n, _, f in probe.LABELS[game])[field]
        indices, values = [], []
        for i, card in enumerate(cards):
            if card.source_game != game or tiers[i] == CardTier.TRAIN:
                continue
            value = extract(card.raw_content)
            if value is not None:
                indices.append(i)
                values.append(str(value))
        values_arr = np.array(values)
        keep_classes = {v for v in set(values_arr) if (values_arr == v).sum() >= 5}
        keep = np.array([v in keep_classes for v in values_arr], dtype=bool)
        out.append((f"{game.value}.{field}", np.array(indices)[keep], values_arr[keep]))
    return out


def effective_rank(x: torch.Tensor) -> float:
    centered = x - x.mean(0)
    energy = torch.linalg.svdvals(centered) ** 2
    shares = energy / energy.sum()
    shares = shares[shares > 0]
    return float(torch.exp(-(shares * shares.log()).sum()))


def score(name, view, x, xs, pool, labels, tiers, slice_sets, out):
    """Append this (model, view)'s rows to the three output tables."""
    summary, by_field, by_game = out
    kinds = {n: kind for n, kind, _, _ in labels}
    linear = {n: probe.linear_probe(x[idx].numpy(), kind, y) for n, kind, idx, y in labels if n != "game"}
    knn = {n: probe.knn_balanced_accuracy(x[idx], y) for n, kind, idx, y in labels if kind == "cls"}
    decks = per_deck_auc(x, pool, lambda i: True)
    train_decks = per_deck_auc(x, pool, lambda i: tiers[i] == CardTier.TRAIN)
    other_decks = per_deck_auc(x, pool, lambda i: tiers[i] != CardTier.TRAIN)
    low, high = bootstrap_interval(decks)
    slice_scores = {n: probe.linear_probe(xs[idx].numpy(), "cls", y) for n, idx, y in slice_sets}
    mean_cos, top_share = probe.anisotropy(x)
    gwent = [v for g, v in decks if g == GameId.GWENT.value]
    summary.append(
        {
            "model": name,
            "view": view,
            "deck_auc": round(mean_of_games(decks), 4),
            "deck_auc_lo95": round(low, 4),
            "deck_auc_hi95": round(high, 4),
            "deck_auc_train_anchors": round(mean_of_games(train_decks), 4),
            "deck_auc_nontrain_anchors": round(mean_of_games(other_decks), 4),
            "deck_auc_gwent": round(float(np.mean(gwent)), 4),
            "linear_cls_bacc": round(float(np.mean([v for k, v in linear.items() if kinds[k] == "cls"])), 4),
            "linear_reg_r2": round(float(np.mean([v for k, v in linear.items() if kinds[k] == "reg"])), 4),
            "slice_linear_cls_bacc": round(float(np.mean(list(slice_scores.values()))), 4),
            "knn_game_bacc": round(knn["game"], 4),
            "knn_field_bacc": round(float(np.mean([v for k, v in knn.items() if k != "game"])), 4),
            "mean_cos": round(mean_cos, 4),
            "top_pc_share": round(top_share, 4),
            "effective_rank": round(effective_rank(x), 2),
        }
    )
    for field, value in linear.items():
        by_field.append(
            {"model": name, "view": view, "field": field, "kind": kinds[field],
             "probe_set": "pool", "linear": round(value, 4),
             "knn": round(knn[field], 4) if field in knn else ""}
        )
    for field, value in slice_scores.items():
        n = len(next(idx for f, idx, _ in slice_sets if f == field))
        by_field.append(
            {"model": name, "view": view, "field": field, "kind": "cls",
             "probe_set": f"slice_nontrain_n{n}", "linear": round(value, 4), "knn": ""}
        )
    games = sorted({g for g, _ in decks})
    for game in games:
        values = [v for g, v in decks if g == game]
        by_game.append(
            {"model": name, "view": view, "game": game, "n_decks": len(values),
             "deck_auc": round(float(np.mean(values)), 4)}
        )
    row = summary[-1]
    print(
        f"{name:30s} {view:8s} deck {row['deck_auc']:.3f} [{low:.3f}-{high:.3f}] "
        f"gwent {row['deck_auc_gwent']:.3f} | lin cls {row['linear_cls_bacc']:.3f} "
        f"slice {row['slice_linear_cls_bacc']:.3f} r2 {row['linear_reg_r2']:.3f} | "
        f"erank {row['effective_rank']:.1f}",
        flush=True,
    )


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.set_num_threads(6)
    pool = probe.build_pool()
    labels = probe.label_sets(pool)
    extra = slice_cards()
    holdout = load_config("single_mixed").plan.holdout
    tiers = [holdout.tier_of(c.nocab_uuid, c.source_game) for c in pool.cards]
    extra_tiers = [holdout.tier_of(c.nocab_uuid, c.source_game) for c in extra]
    slice_sets = slice_labels(extra, extra_tiers)
    print(f"pool {len(pool.cards)} cards, {len(pool.decks)} decks; slice cards {len(extra)}; "
          + ", ".join(f"{f} n={len(i)}" for f, i, _ in slice_sets), flush=True)

    # Every card's frozen features, computed once
    all_cards = {c.nocab_uuid: c for c in [*pool.cards, *extra]}
    order = list(all_cards)
    row_of = {u: i for i, u in enumerate(order)}
    first = run_training.build_model(load_config("single_mixed").model).to(probe.DEVICE)
    if CACHE.exists():
        cached = torch.load(CACHE)
        assert cached["uuids"] == [str(u) for u in order], "card set changed; delete the cache"
        pooled = cached["pooled"]
    else:
        pooled = frozen_features(first, [all_cards[u] for u in order])
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"uuids": [str(u) for u in order], "pooled": pooled}, CACHE)
    print(f"frozen features {tuple(pooled.shape)}", flush=True)

    out: tuple[list, list, list] = ([], [], [])
    reference = pooled[[row_of[c.nocab_uuid] for c in pool.cards]]
    reference_slice = pooled[[row_of[c.nocab_uuid] for c in extra]]
    score("r2_reference_whitened_mean", "frozen", reference, reference_slice,
          pool, labels, tiers, slice_sets, out)

    models: list[tuple[str, object]] = []
    for cell in CELLS:
        models.append((f"r2_{cell}", (cell, None)))
    for shape in ("single", "multi"):
        for seed in CONTROL_SEEDS:
            models.append((f"r2_control_{shape}_s{seed}", (f"{shape}_mixed", seed)))

    for name, (cell, seed) in models:
        config = load_config(cell)
        if seed is not None:
            torch.manual_seed(seed)
        model = run_training.build_model(config.model).to(probe.DEVICE)
        if seed is None:
            state = torch.load(RUNS / cell / "latest" / "encoder.pt", map_location=probe.DEVICE)
            model.load_state_dict(state)
        model.eval()
        if seed is None:
            sample = pool.cards[:64]
            real = model.isolated_embeddings(sample, "fp16")
            use_cache(model, pooled, row_of)
            fast = model.isolated_embeddings(sample, "fp16")
            cos = torch.nn.functional.cosine_similarity(real, fast).min().item()
            print(f"cache check on {cell}: min cosine real vs cached {cos:.5f}", flush=True)
            assert cos > 0.999, "cached frozen features do not reproduce the model"
        use_cache(model, pooled, row_of)
        views = views_of(model, pool.cards)
        slice_views = views_of(model, extra)
        shape_views = ("isolated", "pipeline") if cell.startswith("multi") else ("isolated",)
        for view in shape_views:
            score(name, view, views[view], slice_views[view], pool, labels, tiers, slice_sets, out)
        del model
        torch.cuda.empty_cache()

    RESULTS.mkdir(parents=True, exist_ok=True)
    write(RESULTS / "probe_summary.csv", out[0])
    write(RESULTS / "probe_by_field.csv", out[1])
    write(RESULTS / "probe_deck_by_game.csv", out[2])
    print("wrote", RESULTS / "probe_summary.csv")


if __name__ == "__main__":
    main()
