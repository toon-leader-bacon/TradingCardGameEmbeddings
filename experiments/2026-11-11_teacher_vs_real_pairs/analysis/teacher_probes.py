"""Probes A, B and C (Log, 2026-11-12): why teacher pairs were not faster.

A. The teacher's ceiling: each game's teacher table (the TRAIN-tier
   table main_T trained on, fitted on TRAIN decks only) scored with the
   same deck AUC as the students. Every encoder here is scored on the
   teacher's cards only, so all rows of one split share decks, cards
   and negatives.
B. TRAIN against VALIDATION decks: the same encoders on 500 TRAIN decks
   per game, sampled with the evaluation sampler's seeded order.
C. Text to teacher: ridge and a small MLP from the frozen whitened mean
   (768-d) to the unit-length teacher vectors, 5 folds by card, trained
   on confident cards. Every card gets an out-of-fold prediction; those
   predictions are scored as encoders "text_ridge" and "text_mlp".

Writes results/probe_deck_auc.csv (encoder x split, weighted and per
game) and results/probe_text_to_teacher.csv (per game R^2 and cosine).
Tables and per-deck files go under
data/evaluations/2026-11-11_teacher_vs_real_pairs/probes/.

Run (project root, ROCm venv, GPU):
    PYTHONPATH=. python experiments/2026-11-11_teacher_vs_real_pairs/analysis/teacher_probes.py
"""

import csv
import importlib.util
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.evaluation.intrinsic.deck_retrieval import evaluation_decks as deck_sampling
from src.evaluation.intrinsic.deck_retrieval.deck_retrieval_auc import DeckRetrievalAuc
from src.evaluation.intrinsic.embedding.embed_corpus import embed_corpus
from src.evaluation.intrinsic.embedding.embedding_table import EmbeddingTable
from src.evaluation.intrinsic.embedding.embedding_table_metadata import EmbeddingTableMetadata
from src.schema.game_id import GameId
from src.schema.splits import Split
from src.teacher.teacher_table import TeacherTable

ANALYSIS = Path(__file__).parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load_module("evaluate", ANALYSIS / "evaluate.py")

RUNS = Path("data/runs/2026-11-11_teacher_vs_real_pairs")
OUT = ev.OUT / "probes"
GAMES = ev.TRAINED_GAMES
CHECKPOINTS = {
    "R_r1": RUNS / "main_R" / "round0001", "T_r1": RUNS / "main_T" / "round0001",
    "R_r7": RUNS / "main_R" / "round0007", "T_r7": RUNS / "main_T" / "round0007",
}
SPLITS = (Split.VALIDATION, Split.TRAIN)
FOLDS = 5
MLP_HIDDEN = 1024
MLP_EPOCHS = 200
MLP_PATIENCE = 15
MLP_BATCH = 512

_logger = logging.getLogger("teacher_probes")


class DictEmbedder:
    """A CardEmbedder over a fixed uuid -> vector map."""

    def __init__(self, vectors: dict) -> None:
        self._vectors = vectors
        self.embedding_dim = len(next(iter(vectors.values())))

    def isolated_embeddings(self, cards, precision) -> torch.Tensor:
        return torch.from_numpy(np.stack([self._vectors[c.nocab_uuid] for c in cards])).float()


def teacher_tables() -> dict[GameId, TeacherTable]:
    """The first table main_T's log reuses per game: the TRAIN-tier one
    its teacher_contrastive dojos trained on."""
    pattern = re.compile(r"reusing teacher table (\S+\.teacher\.arrow)")
    tables: dict[GameId, TeacherTable] = {}
    for line in (RUNS / "main_T" / "run.log").read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            table = TeacherTable.open(Path(match.group(1).replace("\\", "/")))
            tables.setdefault(table.source_game, table)
    missing = set(GAMES) - set(tables)
    if missing:
        raise RuntimeError(f"no teacher table in main_T's log for {sorted(g.value for g in missing)}")
    _logger.info("teacher tables: %s", {g.value: len(t.card_uuids) for g, t in tables.items()})
    return tables


def sample_decks(game: GameId, holdout, split: Split) -> list:
    """Up to ev.DECKS_PER_GAME decks of split, in the evaluation sampler's
    seeded order, each with at least min_cards distinct known cards."""
    spec = deck_sampling.EvaluationDeckSpec(count=ev.DECKS_PER_GAME, seed=ev.SEED)
    box = DeckBox.load([DeckBox.default_output_path(game)])
    if split == Split.VALIDATION:
        return list(deck_sampling.sample_evaluation_decks(box, game, holdout, spec))
    keyed = sorted(
        (deck_sampling._ordering_key(spec.seed, uuid), uuid)
        for uuid in box.all_uuids(game)
        if holdout.deck_split_of(uuid, game) == split
    )
    placeholder = CardBinder.unknown_card_uuid(game)
    result = []
    for _, uuid in keyed:
        if len(result) >= spec.count:
            break
        deck = box.get_by_uuid(uuid)
        if deck is None:
            continue
        evaluation_deck = deck_sampling._evaluation_deck_of(deck, placeholder)
        if len(evaluation_deck.card_uuids) >= spec.min_cards:
            result.append(evaluation_deck)
    return result


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_mlp(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> torch.nn.Module:
    """768 -> 1024 -> 128 MLP (about the student head's size) trained on
    1 - cosine, early-stopped on a 10% split of its training cards."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x_train), generator=generator).numpy()
    cut = max(1, len(order) // 10)
    val, fit = order[:cut], order[cut:]
    xs = torch.from_numpy(x_train).float().to(ev.DEVICE)
    ys = torch.from_numpy(y_train).float().to(ev.DEVICE)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(xs.shape[1], MLP_HIDDEN), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(MLP_HIDDEN, ys.shape[1]),
    ).to(ev.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def loss_of(rows) -> torch.Tensor:
        return (1 - torch.nn.functional.cosine_similarity(model(xs[rows]), ys[rows])).mean()

    rng = np.random.default_rng(seed)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(MLP_EPOCHS):
        model.train()
        shuffled = rng.permutation(fit)
        for start in range(0, len(shuffled), MLP_BATCH):
            optimizer.zero_grad()
            loss_of(shuffled[start:start + MLP_BATCH]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_of(val).item()
        if val_loss < best - 1e-4:
            best, best_state, stale = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= MLP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model.eval()


def text_to_teacher(game: GameId, table: TeacherTable, features: dict) -> tuple[dict, dict, list[dict]]:
    """Out-of-fold ridge and MLP predictions of game's unit teacher
    vectors from its frozen features, and per-method scores on the
    confident held-out cards."""
    x = torch.stack([features[u] for u in table.card_uuids]).float().numpy()
    y = unit(table.vectors.astype(np.float64)).astype(np.float32)
    confident = table.confident
    predictions = {"text_ridge": np.zeros_like(y), "text_mlp": np.zeros_like(y)}
    for fold, (train_rows, test_rows) in enumerate(
        KFold(FOLDS, shuffle=True, random_state=ev.SEED).split(x)
    ):
        fit_rows = train_rows[confident[train_rows]]
        ridge = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(x[fit_rows], y[fit_rows])
        predictions["text_ridge"][test_rows] = ridge.predict(x[test_rows])
        mlp = fit_mlp(x[fit_rows], y[fit_rows], seed=fold)
        with torch.no_grad():
            predictions["text_mlp"][test_rows] = mlp(torch.from_numpy(x[test_rows]).to(ev.DEVICE)).cpu().numpy()
    rows = []
    for method, predicted in predictions.items():
        # R^2 on unit-length predictions: the MLP trains on cosine, so its
        # output length is arbitrary; deck AUC uses direction only
        p, t = unit(predicted[confident]), y[confident]
        r2 = 1 - ((t - p) ** 2).sum() / ((t - t.mean(axis=0)) ** 2).sum()
        cosine = float((unit(p) * t).sum(axis=1).mean())
        rows.append({"game": game.value, "method": method, "cards": len(y),
                     "confident_cards": int(confident.sum()), "r2": float(r2), "cosine": cosine})
        _logger.info("%s %s: R^2 %.3f, cosine %.3f", game.value, method, r2, cosine)
    by_uuid = {m: dict(zip(table.card_uuids, p)) for m, p in predictions.items()}
    return by_uuid["text_ridge"], by_uuid["text_mlp"], rows


def build_table(label: str, vectors: dict, cards: list, binder_versions: dict) -> Path:
    path = OUT / "embeddings" / f"{label}.db"
    if not path.exists():
        metadata = EmbeddingTableMetadata(
            encoder_label=label, checkpoint_dir=None, embedding_dim=len(next(iter(vectors.values()))),
            binder_versions=binder_versions, created_at=datetime.now(timezone.utc),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with EmbeddingTable.create(path, metadata) as table:
            embed_corpus(DictEmbedder(vectors), cards, table, batch_size=4096, precision="fp32")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = time.time()
    holdout = ev.load_config(ev.REFERENCE_CONFIG).plan.holdout
    tables = teacher_tables()
    corpus = ev.corpus_cards()
    all_cards = [c for c in corpus if c.source_game in tables]
    covered_uuids = {u for t in tables.values() for u in t.card_uuids}
    cards = [c for c in all_cards if c.nocab_uuid in covered_uuids]
    _logger.info("teacher-covered cards: %d of %d", len(cards), len(all_cards))
    features = ev.frozen_features(corpus)
    binder_versions = {g: CardBinder.load([CardBinder.default_output_path(g)]).version_for(g) for g in GAMES}

    # Encoders as uuid -> vector over the covered cards
    encoders: dict[str, dict] = {
        "teacher": {u: v for t in tables.values() for u, v in zip(t.card_uuids, t.vectors)},
        "frozen_mean": {c.nocab_uuid: features[c.nocab_uuid].numpy() for c in cards},
    }
    # evaluate.py's own check sample, so the cache check is the one it passed
    sample = random.Random(ev.SEED).sample(corpus, ev.CHECK_CARDS)
    for label, directory in CHECKPOINTS.items():
        embedder = ev.build_embedder("checkpoint", directory, features, sample)
        vectors = embedder.isolated_embeddings(cards, "fp16").numpy()
        encoders[label] = dict(zip((c.nocab_uuid for c in cards), vectors))

    # Probe C: text -> teacher, out of fold
    text_rows: list[dict] = []
    encoders["text_ridge"], encoders["text_mlp"] = {}, {}
    for game in GAMES:
        ridge, mlp, rows = text_to_teacher(game, tables[game], features)
        encoders["text_ridge"].update(ridge)
        encoders["text_mlp"].update(mlp)
        text_rows.extend(rows)
    with (ev.RESULTS / "probe_text_to_teacher.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(text_rows[0]))
        writer.writeheader()
        writer.writerows(text_rows)

    # Probes A and B: every encoder on both splits, teacher cards only
    auc_rows: list[dict] = []
    paths = {label: build_table(label, vectors, cards, binder_versions) for label, vectors in encoders.items()}
    for split in SPLITS:
        decks = [d for g in GAMES for d in sample_decks(g, holdout, split)]
        _logger.info("%s: %d decks", split.value, len(decks))
        for label, path in paths.items():
            deck_auc = DeckRetrievalAuc(decks, holdout, seed=ev.SEED)
            with EmbeddingTable.open(path) as table:
                scalars = deck_auc.run(table, OUT / "intrinsic" / split.value / label).scalars
            row = {"encoder": label, "split": split.value,
                   "deck_auc_weighted": ev.weighted_deck_auc(scalars),
                   "n_missing_cards": scalars.get("n_missing_cards"),
                   **{f"deck_auc.{g.value}": scalars.get(f"deck_auc.{g.value}") for g in GAMES}}
            auc_rows.append(row)
            _logger.info("%s %s: weighted deck AUC %.4f", split.value, label, row["deck_auc_weighted"])
    with (ev.RESULTS / "probe_deck_auc.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(auc_rows[0]))
        writer.writeheader()
        writer.writerows(auc_rows)
    _logger.info("done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
