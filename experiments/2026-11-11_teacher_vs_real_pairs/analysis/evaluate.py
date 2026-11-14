"""Offline evaluation of every encoder in this experiment (Method, step 3).

Each encoder is scored the same way:
1. same-deck retrieval AUC (src/evaluation DeckRetrievalAuc) on a fixed,
   seeded sample of VALIDATION decks per game, the 15 trained games plus
   Gwent; per-game scalars, bootstrap CIs, train vs held-out-tier anchors,
   and per_deck.csv for paired comparison (compare.py);
2. effective rank and mean pairwise cosine of the whole table;
3. the card-field probe: 5-fold logistic balanced accuracy ("linear-cls")
   and ridge R^2 on raw_content fields, from the pooling probe's label
   set (experiments/2026-10-10_pooling_probe/analysis/probe.py), on up to
   PROBE_CARDS cards per game.

The frozen part of every model (ModernBERT, token whitening, mean
pooling) is identical across encoders, so it is computed once per card and
saved; each encoder runs only its head. Each trained encoder is checked:
the cached path against the real model on 64 cards.

Encoders (repeatable):
    --checkpoint LABEL=<run>/<round dir>    trained head (encoder.pt; the run's
                                            run_config.yaml sits in <run>/)
    --untrained LABEL=<run config yaml>     build_model(spec, seed=plan.seed)
    --frozen-mean                           the head-free 768-d whitened mean

Writes data/evaluations/2026-11-11_teacher_vs_real_pairs/ (frozen
features, embedding tables, analysis outputs) and one row per encoder into
results/eval_summary.csv and results/deck_auc_by_game.csv (rows for
re-scored labels are replaced). Re-runnable: finished tables are reused.

Run (project root, ROCm venv, GPU):
    PYTHONPATH=. python experiments/2026-11-11_teacher_vs_real_pairs/analysis/evaluate.py \
        --frozen-mean --untrained untrained=<config> --checkpoint T_r3=<run>/train_round0002
"""

import argparse
import csv
import dataclasses
import importlib.util
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.encoder_model.card_serialization import serialize_card_to_json
from src.evaluation.intrinsic.deck_retrieval.deck_retrieval_auc import DeckRetrievalAuc
from src.evaluation.intrinsic.deck_retrieval.evaluation_decks import (
    EvaluationDeck,
    EvaluationDeckSpec,
    sample_evaluation_decks,
)
from src.evaluation.intrinsic.effective_rank import EffectiveRank
from src.evaluation.intrinsic.embedding.embed_corpus import embed_corpus
from src.evaluation.intrinsic.embedding.embedding_table import EmbeddingTable
from src.evaluation.intrinsic.embedding.embedding_table_metadata import EmbeddingTableMetadata
from src.schema.card import GenericCard
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.dojo_catalog import DOJO_CATALOG
from src.training.run_config import parse_run_config, read_config_document


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_training = _load_module("run_training", "scripts/run_training.py")
probe = _load_module("probe", "experiments/2026-10-10_pooling_probe/analysis/probe.py")

EXPERIMENT = "2026-11-11_teacher_vs_real_pairs"
OUT = Path("data/evaluations") / EXPERIMENT
RESULTS = Path("experiments") / EXPERIMENT / "results"
FEATURES = OUT / "frozen_features.pt"
REFERENCE_CONFIG = Path("experiments") / EXPERIMENT / "configs" / "pilot_T_lr3e-04.yaml"

TRAINED_GAMES = (
    GameId.MTG, GameId.YUGIOH, GameId.POKEMON, GameId.HEARTHSTONE,
    GameId.FLESH_AND_BLOOD, GameId.DIGIMON, GameId.ONE_PIECE,
    GameId.SLAY_THE_SPIRE_2, GameId.STAR_WARS_UNLIMITED,
    GameId.DRAGON_BALL_MASTERS, GameId.GUNDAM, GameId.DRAGON_BALL_FUSION_WORLD,
    GameId.LORCANA, GameId.BATTLE_SPIRITS_SAGA, GameId.DOMINION,
)
CORPUS_GAMES = (*TRAINED_GAMES, GameId.GWENT)
# round(sqrt(teacher TRAIN anchors)), the diet table (make_configs.py)
DIET_WEIGHTS = {
    GameId.MTG: 160, GameId.YUGIOH: 106, GameId.POKEMON: 78, GameId.HEARTHSTONE: 66,
    GameId.FLESH_AND_BLOOD: 62, GameId.DIGIMON: 54, GameId.ONE_PIECE: 40,
    GameId.SLAY_THE_SPIRE_2: 21, GameId.STAR_WARS_UNLIMITED: 20,
    GameId.DRAGON_BALL_MASTERS: 19, GameId.GUNDAM: 19,
    GameId.DRAGON_BALL_FUSION_WORLD: 18, GameId.LORCANA: 16,
    GameId.BATTLE_SPIRITS_SAGA: 16, GameId.DOMINION: 12,
}
DECKS_PER_GAME = 500
PROBE_CARDS = 2000
SEED = 0
DEVICE = "cuda"
TEXT_BATCH = 64
CHECK_CARDS = 64

_logger = logging.getLogger("evaluate")


def load_config(path: Path):
    return parse_run_config(read_config_document(path), tuple(DOJO_CATALOG))


def corpus_cards() -> list[GenericCard]:
    """Every known card of the corpus games, in game then uuid order."""
    cards: list[GenericCard] = []
    for game in CORPUS_GAMES:
        binder = CardBinder.load([CardBinder.default_output_path(game)])
        unknown = CardBinder.unknown_card_uuid(game)
        game_cards = [c for c in binder.all_cards(game) if c.nocab_uuid != unknown]
        cards.extend(sorted(game_cards, key=lambda c: str(c.nocab_uuid)))
    return cards


def frozen_features(cards: list[GenericCard]) -> dict:
    """nocab_uuid -> whitened mean-pooled token states (768,), computed
    once with the reference config's frozen stages and saved."""
    if FEATURES.exists():
        saved = torch.load(FEATURES)
        by_uuid = dict(zip(saved["uuids"], saved["pooled"]))
        if all(str(c.nocab_uuid) in by_uuid for c in cards):
            return {c.nocab_uuid: by_uuid[str(c.nocab_uuid)] for c in cards}
        _logger.info("saved features miss cards; recomputing")
    spec = dataclasses.replace(load_config(REFERENCE_CONFIG).model, feature_cache=None)
    pipeline = run_training.build_model(spec, seed=SEED).to(DEVICE).eval().card_pipeline
    texts = {c.nocab_uuid: serialize_card_to_json(c) for c in cards}
    order = sorted(cards, key=lambda c: len(texts[c.nocab_uuid]))
    result: dict = {}
    started = time.time()
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16):
        for start in range(0, len(order), TEXT_BATCH):
            chunk = order[start : start + TEXT_BATCH]
            encoding = pipeline.text_encoder.encode([texts[c.nocab_uuid] for c in chunk])
            if pipeline.enrichment is not None:
                encoding = pipeline.enrichment(encoding)
            for card, row in zip(chunk, pipeline.pooling(encoding).float().cpu()):
                result[card.nocab_uuid] = row
            if (start // TEXT_BATCH) % 200 == 0:
                _logger.info("frozen features %d/%d (%.0fs)", start, len(order), time.time() - started)
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"uuids": [str(u) for u in result], "pooled": torch.stack(list(result.values()))}, FEATURES)
    return result


class CachedHeadEmbedder:
    """A CardEmbedder (embed_corpus's Protocol) that runs only a model's
    embedding head over the saved frozen features; with no head, it returns
    the frozen features themselves (the whitened-mean reference)."""

    def __init__(self, features: dict, head: torch.nn.Module | None) -> None:
        self._features = features
        self._head = head
        self.embedding_dim = 768 if head is None else head.output_dim

    def isolated_embeddings(self, cards, precision) -> torch.Tensor:
        x = torch.stack([self._features[c.nocab_uuid] for c in cards])
        if self._head is None:
            return x.float()
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16):
            return self._head(x.to(DEVICE)).float().cpu()


def build_embedder(kind: str, source: Path | None, features: dict, sample: list) -> CachedHeadEmbedder:
    """The embedder for one encoder; a trained one is checked against its
    real model on `sample`."""
    if kind == "frozen_mean":
        return CachedHeadEmbedder(features, None)
    config_path = source if kind == "untrained" else source.parent / "run_config.yaml"
    config = load_config(config_path)
    spec = dataclasses.replace(config.model, feature_cache=None)
    model = run_training.build_model(spec, seed=config.plan.seed).to(DEVICE)
    if kind == "checkpoint":
        model.load_state_dict(torch.load(source / "encoder.pt", map_location=DEVICE))
    model.eval()
    embedder = CachedHeadEmbedder(features, model.card_pipeline.embedding_head)
    real = model.isolated_embeddings(sample, "fp16")
    fast = embedder.isolated_embeddings(sample, "fp16")
    cosine = torch.nn.functional.cosine_similarity(real, fast).min().item()
    _logger.info("cache check: min cosine real vs cached %.5f", cosine)
    if cosine < 0.999:
        raise RuntimeError("cached frozen features do not reproduce the model")
    return embedder


def evaluation_decks(holdout: HoldoutSpec) -> list[EvaluationDeck]:
    decks: list[EvaluationDeck] = []
    spec = EvaluationDeckSpec(count=DECKS_PER_GAME, seed=SEED)
    for game in CORPUS_GAMES:
        decks.extend(sample_evaluation_decks(DeckBox.load([DeckBox.default_output_path(game)]), game, holdout, spec))
    return decks


def probe_scores(vectors: np.ndarray, cards: list[GenericCard]) -> dict[str, float]:
    """linear-cls (mean balanced accuracy over class fields) and linear-reg
    (mean R^2 over numeric fields) on up to PROBE_CARDS cards per game,
    the same seeded cards for every encoder; plus each field's score."""
    rng = random.Random(SEED)
    scores: dict[str, float] = {}
    by_game: dict[GameId, list[int]] = {}
    for row, card in enumerate(cards):
        by_game.setdefault(card.source_game, []).append(row)
    for game, specs in probe.LABELS.items():
        rows = by_game.get(game, [])
        rows = sorted(rng.sample(rows, min(PROBE_CARDS, len(rows))))
        for name, kind, extract in specs:
            pairs = [(r, extract(cards[r].raw_content)) for r in rows]
            pairs = [(r, v) for r, v in pairs if v is not None]
            if kind == "cls":
                labels = np.array([str(v) for _, v in pairs])
                keep = {v for v in set(labels) if (labels == v).sum() >= 5}
                pairs = [p for p, label in zip(pairs, labels) if label in keep]
                if len(keep) < 2:
                    continue
                y = np.array([str(v) for _, v in pairs])
            else:
                if len(pairs) < 30:
                    continue
                y = np.array([v for _, v in pairs], dtype=float)
            x = vectors[[r for r, _ in pairs]]
            scores[f"{kind}.{game.value}.{name}"] = probe.linear_probe(x, kind, y)
    for kind in ("cls", "reg"):
        values = [v for k, v in scores.items() if k.startswith(f"{kind}.") and not k.startswith(f"{kind}.gwent")]
        scores[f"linear_{kind}"] = float(np.mean(values))
    return scores


def weighted_deck_auc(scalars: dict) -> float:
    total = sum(DIET_WEIGHTS.values())
    return sum(w * scalars[f"deck_auc.{g.value}"] for g, w in DIET_WEIGHTS.items()) / total


def mean_cosine(vectors: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    rows = vectors[rng.choice(len(vectors), size=min(5000, len(vectors)), replace=False)]
    unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    sims = unit @ unit.T
    n = len(unit)
    return float((sims.sum() - n) / (n * (n - 1)))


def write_rows(path: Path, rows: list[dict], key: str = "encoder") -> None:
    """Replace rows with the same encoder label; keep the rest."""
    labels = {r[key] for r in rows}
    old: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            old = [r for r in csv.DictReader(f) if r[key] not in labels]
    merged = old + rows
    fields = list(dict.fromkeys(k for r in merged for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)


def evaluate(label: str, embedder: CachedHeadEmbedder, cards: list, binder_versions: dict,
             deck_auc: DeckRetrievalAuc, source: str) -> None:
    table_path = OUT / "embeddings" / f"{label}.db"
    started = time.time()
    if not table_path.exists():
        metadata = EmbeddingTableMetadata(
            encoder_label=label, checkpoint_dir=Path(source) if source else None,
            embedding_dim=embedder.embedding_dim, binder_versions=binder_versions,
            created_at=datetime.now(timezone.utc),
        )
        with EmbeddingTable.create(table_path, metadata) as table:
            embed_corpus(embedder, cards, table, batch_size=1024, precision="fp16")
    vectors = embedder.isolated_embeddings(cards, "fp16").numpy()
    out = OUT / "intrinsic" / label
    with EmbeddingTable.open(table_path) as table:
        deck = deck_auc.run(table, out / deck_auc.name).scalars
        rank = EffectiveRank().run(table, out / "effective_rank").scalars
    probes = probe_scores(vectors, cards)
    summary = {
        "encoder": label, "source": source,
        "deck_auc_weighted": weighted_deck_auc(deck),
        "deck_auc_unweighted_trained": float(np.mean([deck[f"deck_auc.{g.value}"] for g in TRAINED_GAMES])),
        "deck_auc_gwent": deck.get("deck_auc.gwent"),
        "deck_auc_heldout_anchor": deck.get("deck_auc.heldout_anchor"),
        "deck_auc_train_anchor": deck.get("deck_auc.train_anchor"),
        "linear_cls": probes["linear_cls"], "linear_reg": probes["linear_reg"],
        "effective_rank": rank.get("effective_rank"), "mean_cosine": mean_cosine(vectors),
        **{f"probe.{k}": v for k, v in probes.items() if k.count(".") >= 2},
    }
    by_game = [
        {"encoder": label, "game": g.value, "deck_auc": deck.get(f"deck_auc.{g.value}"),
         "lo95": deck.get(f"deck_auc.{g.value}_lo95"), "hi95": deck.get(f"deck_auc.{g.value}_hi95")}
        for g in CORPUS_GAMES
    ]
    write_rows(RESULTS / "eval_summary.csv", [summary])
    write_rows(RESULTS / "deck_auc_by_game.csv", by_game)
    _logger.info("%s: weighted deck AUC %.4f, linear-cls %.4f, rank %.1f (%.0fs)",
                 label, summary["deck_auc_weighted"], summary["linear_cls"],
                 summary["effective_rank"] or float("nan"), time.time() - started)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=DIR")
    parser.add_argument("--untrained", action="append", default=[], metavar="LABEL=CONFIG")
    parser.add_argument("--frozen-mean", action="store_true")
    args = parser.parse_args()

    holdout = load_config(REFERENCE_CONFIG).plan.holdout
    cards = corpus_cards()
    _logger.info("corpus: %d cards over %d games", len(cards), len(CORPUS_GAMES))
    features = frozen_features(cards)
    binder_versions = {
        g: CardBinder.load([CardBinder.default_output_path(g)]).version_for(g) for g in CORPUS_GAMES
    }
    deck_auc = DeckRetrievalAuc(evaluation_decks(holdout), holdout, seed=SEED)
    sample = random.Random(SEED).sample(cards, CHECK_CARDS)

    jobs: list[tuple[str, str, Path | None]] = []
    if args.frozen_mean:
        jobs.append(("frozen_mean", "frozen_mean", None))
    for kind, items in (("untrained", args.untrained), ("checkpoint", args.checkpoint)):
        for item in items:
            label, _, path = item.partition("=")
            jobs.append((label, kind, Path(path)))
    for label, kind, source in jobs:
        embedder = build_embedder(kind, source, features, sample)
        evaluate(label, embedder, cards, binder_versions, deck_auc, str(source or ""))


if __name__ == "__main__":
    main()
