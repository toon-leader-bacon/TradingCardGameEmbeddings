"""Probes C2 and E (Log, 2026-11-12): head capacity, and the training
signal isolated, offline on the cached frozen features.

C2. Per game, ResidualMlpEmbeddingHead(768, 128, width, 3) fitted to the
    unit teacher vectors (1 - cosine) on all confident cards, a fixed
    C2_STEPS, no held-out data; predictions scored with the probe A deck
    AUC (VALIDATION decks, teacher cards). Widths C2_WIDTHS.
E.  One shared ResidualMlpEmbeddingHead(768, 256, width, 3) over the 15
    games, trained like the main runs (AdamW 3e-4, 2k warm-up, cosine to
    0.1 at 80k) on identical batches: a diet-weighted game, 64 confident
    anchors, 7 of each anchor's teacher top-32. Only the target differs:
    "pairs" (uniform over the batch cards in a card's top-32) or "kl"
    (softmax of the teacher's cosines). Scored with evaluate.py's full
    evaluation at E_EVAL_STEPS.

Writes results/probe_c2_capacity.csv and rows E_* in
results/eval_summary.csv (via evaluate.py), plus results/probe_e_curves.csv.

Run (project root, ROCm venv, GPU):
    PYTHONPATH=. python experiments/2026-11-11_teacher_vs_real_pairs/analysis/distill_probes.py [--only c2|e]
"""

import argparse
import csv
import importlib.util
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data_refinement.card_binder.card_binder import CardBinder
from src.encoder_model.embedding_head import ResidualMlpEmbeddingHead
from src.evaluation.intrinsic.deck_retrieval.deck_retrieval_auc import DeckRetrievalAuc
from src.evaluation.intrinsic.embedding.embedding_table import EmbeddingTable
from src.schema.splits import Split

ANALYSIS = Path(__file__).parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tp = _load_module("teacher_probes", ANALYSIS / "teacher_probes.py")
ev = tp.ev

DEVICE = ev.DEVICE
K = 32
C2_WIDTHS = (512, 1024, 2048)
C2_STEPS = 4000
C2_BATCH = 512
E_ARMS = {"E_pairs": ("pairs", 512), "E_kl": ("kl", 512), "E_kl_wide": ("kl", 2048),
          # main_T's batch shape: 256 random (anchor, one top-32 neighbour) pairs,
          # single-positive InfoNCE at the catalog's 0.07 (SingleCardInfoNCELoss)
          "E_t_replica": ("single", 512),
          # width control for E_kl_wide
          "E_pairs_wide": ("pairs", 2048)}
SINGLE_PAIRS, SINGLE_TEMPERATURE = 256, 0.07
E_STEPS = 80_000
E_EVAL_STEPS = (10_000, 20_000, 40_000, 80_000)
# Also scored on teacher cards only (probe A's view), to compare with A and C2
E_COVERED_STEPS = (10_000, 80_000)
E_ANCHORS, E_PER_ANCHOR = 64, 7
TEMPERATURE = 0.05
LR, WEIGHT_DECAY, WARMUP, FLOOR = 3e-4, 0.01, 2000, 0.1

_logger = logging.getLogger("distill_probes")


def head_of(output_dim: int, width: int, seed: int = 0) -> ResidualMlpEmbeddingHead:
    torch.manual_seed(seed)
    return ResidualMlpEmbeddingHead(768, output_dim, hidden_dim=width, num_blocks=3).to(DEVICE)


def parameter_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class GameData:
    """One game's confident teacher cards on the GPU: frozen features,
    unit teacher vectors, and each card's top-K teacher neighbours (row
    indices into the same arrays)."""

    def __init__(self, table, features: dict) -> None:
        keep = np.flatnonzero(table.confident)
        self.uuids = [table.card_uuids[i] for i in keep]
        self.x = torch.stack([features[u] for u in self.uuids]).float().to(DEVICE)
        self.t = F.normalize(torch.from_numpy(table.vectors[keep]).float().to(DEVICE), dim=1)
        self.neighbours = self._top_k()

    def _top_k(self) -> torch.Tensor:
        k = min(K, len(self.t) - 1)
        rows = []
        for start in range(0, len(self.t), 4096):
            sims = self.t[start:start + 4096] @ self.t.T
            sims[torch.arange(len(sims)), torch.arange(start, start + len(sims))] = -2
            rows.append(sims.topk(k, dim=1).indices)
        return torch.cat(rows)


def lr_factor(step: int) -> float:
    if step < WARMUP:
        return (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, E_STEPS - WARMUP)
    return FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def batch_rows(game: GameData, generator: torch.Generator, kind: str = "pairs") -> torch.Tensor:
    if kind == "single":
        anchors = torch.randint(len(game.t), (SINGLE_PAIRS,), generator=generator, device=DEVICE)
        picks = torch.randint(game.neighbours.shape[1], (SINGLE_PAIRS,), generator=generator, device=DEVICE)
        return torch.stack([anchors, game.neighbours[anchors, picks]], dim=1).reshape(-1)
    anchors = torch.randint(len(game.t), (E_ANCHORS,), generator=generator, device=DEVICE)
    picks = torch.randint(game.neighbours.shape[1], (E_ANCHORS, E_PER_ANCHOR), generator=generator, device=DEVICE)
    mates = game.neighbours[anchors.unsqueeze(1), picks]
    return torch.cat([anchors.unsqueeze(1), mates], dim=1).reshape(-1)


def batch_loss(kind: str, head, game: GameData, rows: torch.Tensor) -> torch.Tensor:
    s = F.normalize(head(game.x[rows]).float(), dim=1)
    same = rows.unsqueeze(0) == rows.unsqueeze(1)
    if kind == "single":
        # rows interleave (anchor, partner): each card's one positive is
        # its partner; every other card is a negative, neighbours included
        partner = torch.arange(len(rows), device=DEVICE) ^ 1
        logits = (s @ s.T / SINGLE_TEMPERATURE).masked_fill(same, -1e4)
        return F.cross_entropy(logits, partner)
    logits = (s @ s.T / TEMPERATURE).masked_fill(same, -1e4)
    if kind == "kl":
        t = game.t[rows]
        target = F.softmax((t @ t.T / TEMPERATURE).masked_fill(same, -1e4), dim=1)
    else:
        member = (game.neighbours[rows].unsqueeze(1) == rows.view(1, -1, 1)).any(dim=2) & ~same
        keep = member.any(dim=1)
        target = member[keep].float() / member[keep].sum(dim=1, keepdim=True)
        logits = logits[keep]
    return F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")


# ---------------------------------------------------------------- C2


def run_c2(tables, features, cards, binder_versions, holdout) -> None:
    decks = [d for g in tp.GAMES for d in tp.sample_decks(g, holdout, Split.VALIDATION)]
    rows = []
    for width in C2_WIDTHS:
        predictions: dict = {}
        fit_cosines = {}
        for game in tp.GAMES:
            table = tables[game]
            data = GameData(table, features)
            head = head_of(table.dim, width)
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
            generator = torch.Generator(device=DEVICE).manual_seed(0)
            head.train()
            for _ in range(C2_STEPS):
                idx = torch.randint(len(data.t), (C2_BATCH,), generator=generator, device=DEVICE)
                loss = (1 - F.cosine_similarity(head(data.x[idx]), data.t[idx])).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            head.eval()
            x_all = torch.stack([features[u] for u in table.card_uuids]).float().to(DEVICE)
            with torch.no_grad():
                out = head(x_all).float().cpu().numpy()
                fit_cosines[game.value] = float(F.cosine_similarity(head(data.x), data.t).mean())
            predictions.update(zip(table.card_uuids, out))
        label = f"c2_w{width}"
        params = parameter_count(head_of(128, width))
        path = tp.build_table(label, predictions, cards, binder_versions)
        with EmbeddingTable.open(path) as table_db:
            scalars = DeckRetrievalAuc(decks, holdout, seed=ev.SEED).run(
                table_db, tp.OUT / "intrinsic" / "validation" / label).scalars
        row = {"encoder": label, "width": width, "params_per_game_head": params,
               "deck_auc_weighted": ev.weighted_deck_auc(scalars),
               "fit_cosine_mean": float(np.mean(list(fit_cosines.values()))),
               **{f"deck_auc.{g.value}": scalars.get(f"deck_auc.{g.value}") for g in tp.GAMES},
               **{f"fit_cosine.{g}": c for g, c in fit_cosines.items()}}
        rows.append(row)
        _logger.info("C2 width %d (%d params/game): deck AUC %.4f, in-sample cosine %.3f",
                     width, params, row["deck_auc_weighted"], row["fit_cosine_mean"])
    with (ev.RESULTS / "probe_c2_capacity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- E


def scored_labels() -> set[str]:
    path = ev.RESULTS / "eval_summary.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {r["encoder"] for r in csv.DictReader(f)}


def score_covered(label: str, head, features, cards, binder_versions, holdout, decks) -> dict:
    """label's deck AUC on teacher cards and VALIDATION decks (probe A)."""
    x = torch.stack([features[c.nocab_uuid] for c in cards]).float().to(DEVICE)
    with torch.no_grad():
        vectors = torch.cat([head(x[i:i + 8192]).float() for i in range(0, len(x), 8192)]).cpu().numpy()
    path = tp.build_table(label, dict(zip((c.nocab_uuid for c in cards), vectors)), cards, binder_versions)
    with EmbeddingTable.open(path) as table:
        scalars = DeckRetrievalAuc(decks, holdout, seed=ev.SEED).run(
            table, tp.OUT / "intrinsic" / "validation" / label).scalars
    return {"encoder": label, "split": "validation", "deck_auc_weighted": ev.weighted_deck_auc(scalars),
            **{f"deck_auc.{g.value}": scalars.get(f"deck_auc.{g.value}") for g in tp.GAMES}}


def run_e(tables, features, corpus, cards, binder_versions, holdout, arms=None) -> None:
    games = {g: GameData(tables[g], features) for g in tp.GAMES}
    covered_decks = [d for g in tp.GAMES for d in tp.sample_decks(g, holdout, Split.VALIDATION)]
    already = scored_labels()
    covered_rows = []
    (tp.OUT / "heads").mkdir(parents=True, exist_ok=True)
    weights = torch.tensor([ev.DIET_WEIGHTS[g] for g in tp.GAMES], dtype=torch.float)
    deck_auc = DeckRetrievalAuc(ev.evaluation_decks(holdout), holdout, seed=ev.SEED)
    curve = []
    for label, (kind, width) in E_ARMS.items():
        if arms and label not in arms:
            continue
        head = head_of(256, width)
        _logger.info("%s: %s target, width %d, %d params", label, kind, width, parameter_count(head))
        optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        game_draws = torch.multinomial(weights, E_STEPS, replacement=True,
                                       generator=torch.Generator().manual_seed(0))
        started, running = time.time(), []
        for step in range(E_STEPS):
            head.train()
            game = games[tp.GAMES[int(game_draws[step])]]
            loss = batch_loss(kind, head, game, batch_rows(game, generator, kind))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 50 == 0:
                running.append(loss.item())
            if step + 1 in E_EVAL_STEPS:
                head.eval()
                train_seconds = time.time() - started
                step_label = f"{label}_s{(step + 1) // 1000}k"
                torch.save(head.state_dict(), tp.OUT / "heads" / f"{step_label}.pt")
                if step_label not in already:
                    ev.evaluate(step_label, ev.CachedHeadEmbedder(features, head), corpus,
                                binder_versions, deck_auc, f"distill_probes:{label}")
                if step + 1 in E_COVERED_STEPS:
                    covered_rows.append(score_covered(f"{step_label}_covered", head, features, cards,
                                                      binder_versions, holdout, covered_decks))
                    _logger.info("%s on teacher cards: %.4f", step_label, covered_rows[-1]["deck_auc_weighted"])
                curve.append({"arm": label, "step": step + 1, "train_seconds": train_seconds,
                              "mean_loss_recent": float(np.mean(running[-40:]))})
                _logger.info("%s step %d: %.0f s of training, recent loss %.4f",
                             label, step + 1, train_seconds, curve[-1]["mean_loss_recent"])
                started = time.time() - train_seconds
    if covered_rows:
        ev.write_rows(ev.RESULTS / "probe_e_covered.csv", covered_rows)
    path = ev.RESULTS / "probe_e_curves.csv"
    done = {r["arm"] for r in curve}
    old = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            old = [r for r in csv.DictReader(f) if r["arm"] not in done]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(old + curve)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("c2", "e"))
    parser.add_argument("--arms", nargs="*", help="E arms to run (default: all)")
    args = parser.parse_args()
    holdout = ev.load_config(ev.REFERENCE_CONFIG).plan.holdout
    tables = tp.teacher_tables()
    corpus = ev.corpus_cards()
    features = ev.frozen_features(corpus)
    covered = {u for t in tables.values() for u in t.card_uuids}
    cards = [c for c in corpus if c.nocab_uuid in covered]
    binder_versions = {g: CardBinder.load([CardBinder.default_output_path(g)]).version_for(g)
                       for g in ev.CORPUS_GAMES}
    if args.only in (None, "c2"):
        run_c2(tables, features, cards, binder_versions, holdout)
    if args.only in (None, "e"):
        run_e(tables, features, corpus, cards, binder_versions, holdout, args.arms)


if __name__ == "__main__":
    main()
