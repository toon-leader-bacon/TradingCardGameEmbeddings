"""The evaluation module's intrinsic suite on every cell and control, plus
learning curves of the fresh-head runs.

Uses src/evaluation as built (see its README), so the outputs have the
module's standard layout and compare across experiments:

1. one EmbeddingTable per encoder over every card of every game
   (select_corpus, embed_corpus; each card embedded alone, the isolated
   view);
2. EffectiveRank on the whole table (unlabeled);
3. ClusterAgreement, LabelCompactness and ProjectionPlot for each label
   source, PerLabelCap(500), seed 0;
4. plot_learning_curves over the fresh-head runs' rounds CSVs
   (extrinsic_held_out.py), one chart per held-out dojo.

Label sources (LABEL_SOURCES below):
- game (GameLabels) and holdout_tier (HoldoutTierLabels, the runs' holdout);
- fields whose dojos are held out in every cell (rarity tier across games,
  Hearthstone class, Flesh and Blood pitch, Slay the Spire 2 card type);
- fields trained in the metric and mixed cells (MTG card type, Flesh and
  Blood class, Pokemon stage);
- Gwent faction (the game no dojo trains).

Encoders: the six cells' latest/encoder.pt, and the untrained controls
(build_model under torch.manual_seed(s), s = 0, 1, 2, per model shape), as
in probe_cells.py.

Speed: the frozen part (ModernBERT, token whitening, mean pooling) is the
same in every encoder, so it is computed once per card, saved, and each
encoder runs only its head (and group attention, for multi-card). Each
trained cell checks the cached path against the real model on 64 cards.

Writes data/evaluations/2026-10-11_model_by_diet_r2/ (embeddings/,
intrinsic/, extrinsic/plots/, frozen_features.pt). Re-runnable: finished
tables and analyses are skipped. intrinsic_report.py summarizes it.

Run (from the project root, ROCm venv, GPU):
    PYTHONPATH=. python experiments/2026-10-11_model_by_diet_r2/analysis/intrinsic_suite.py
"""

import importlib.util
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.encoder_model.card_serialization import serialize_card_to_json
from src.evaluation.analyses.card_sample import PerLabelCap
from src.evaluation.analyses.cluster_agreement import ClusterAgreement
from src.evaluation.analyses.effective_rank import EffectiveRank
from src.evaluation.analyses.embedding_analysis import EmbeddingAnalysis
from src.evaluation.analyses.label_compactness import LabelCompactness
from src.evaluation.analyses.projection_plot import ProjectionPlot
from src.evaluation.embedding.embed_corpus import embed_corpus
from src.evaluation.embedding.embedding_table import EmbeddingTable
from src.evaluation.embedding.embedding_table_metadata import EmbeddingTableMetadata
from src.evaluation.extrinsic.learning_curves import plot_learning_curves
from src.evaluation.labels.card_labels import CardLabels, GameLabels, HoldoutTierLabels
from src.evaluation.labels.metric_parquet_labels import MetricParquetLabels
from src.evaluation.select_corpus import CorpusSpec, select_corpus
from src.schema.game_id import GameId
from src.training.dojo_catalog import DOJO_CATALOG
from src.training.run_config import parse_run_config, read_config_document

_spec = importlib.util.spec_from_file_location("run_training", "scripts/run_training.py")
assert _spec is not None and _spec.loader is not None
run_training = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_training)

RUNS = Path("data/runs/2026-10-11_model_by_diet_r2")
OUT = Path("data/evaluations/2026-10-11_model_by_diet_r2")
FEATURES = OUT / "frozen_features.pt"
EXTRINSIC = RUNS / "extrinsic" / "extrinsic"
CELLS = (
    "single_contrastive",
    "single_metric",
    "single_mixed",
    "multi_contrastive",
    "multi_metric",
    "multi_mixed",
)
CONTROL_SEEDS = (0, 1, 2)
EXTRINSIC_ENCODERS = (
    "control_single_s0",
    "single_contrastive",
    "single_metric",
    "single_mixed",
    "control_multi_s0",
    "multi_contrastive",
    "multi_metric",
    "multi_mixed",
)
METRICS = Path("data/metrics")
# (label source name, parquet) - held out in every cell, trained in the
# metric and mixed cells, and the unseen game
METRIC_LABELS = (
    ("heldout_rarity_tier", "cross_game/rarity_tier"),
    ("heldout_hs_class", "hearthstonejson/class_mask"),
    ("heldout_fab_pitch", "cardvault_fabtcg/pitch_mask"),
    ("heldout_sts2_card_type", "spire_codex/card_type_mask"),
    ("trained_mtg_card_type", "scryfall/card_type_mask"),
    ("trained_fab_class", "cardvault_fabtcg/class_mask"),
    ("trained_pokemon_stage", "pokemon_tcg/stage_mask"),
    ("unseen_gwent_faction", "gwent_one/faction_mask"),
)
PER_LABEL_CAP = 500
SEED = 0
DEVICE = "cuda"
TEXT_BATCH = 64
EMBED_BATCH = 1024

_logger = logging.getLogger("intrinsic_suite")


def load_config(cell: str):
    return parse_run_config(
        read_config_document(RUNS / cell / "run_config.yaml"), tuple(DOJO_CATALOG)
    )


def frozen_features(model, cards: list) -> dict:
    """nocab_uuid -> whitened mean-pooled token states (768,), computed once
    and saved. Cards are batched by serialized length to cut padding."""
    if FEATURES.exists():
        saved = torch.load(FEATURES)
        by_text = dict(zip(saved["uuids"], saved["pooled"]))
        if all(str(c.nocab_uuid) in by_text for c in cards):
            return {c.nocab_uuid: by_text[str(c.nocab_uuid)] for c in cards}
        _logger.info("saved features miss cards; recomputing")
    pipeline = model.card_pipeline
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
            pooled = pipeline.pooling(encoding).float().cpu()
            for card, row in zip(chunk, pooled):
                result[card.nocab_uuid] = row
            if (start // TEXT_BATCH) % 100 == 0:
                _logger.info("frozen features %d/%d (%.0fs)", start, len(order), time.time() - started)
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"uuids": [str(u) for u in result],
                "pooled": torch.stack(list(result.values()))}, FEATURES)
    return result


def use_features(model, features: dict) -> None:
    """Make model.card_pipeline.embed read the saved frozen features."""

    def embed(self, cards):
        x = torch.stack([features[c.nocab_uuid] for c in cards]).to(DEVICE)
        return self.embedding_head(x)

    model.card_pipeline.embed = MethodType(embed, model.card_pipeline)


def encoders() -> list[tuple[str, str, int | None]]:
    """(label, config cell, control seed or None for the trained weights)."""
    rows: list[tuple[str, str, int | None]] = [(cell, cell, None) for cell in CELLS]
    for shape in ("single", "multi"):
        rows += [(f"control_{shape}_s{s}", f"{shape}_mixed", s) for s in CONTROL_SEEDS]
    return rows


def build_encoder(cell: str, seed: int | None, features: dict, sample: list):
    if seed is not None:
        torch.manual_seed(seed)
    model = run_training.build_model(load_config(cell).model).to(DEVICE)
    if seed is None:
        model.load_state_dict(torch.load(RUNS / cell / "latest" / "encoder.pt", map_location=DEVICE))
    model.eval()
    if seed is None:
        real = model.isolated_embeddings(sample, "fp16")
        use_features(model, features)
        fast = model.isolated_embeddings(sample, "fp16")
        cos = torch.nn.functional.cosine_similarity(real, fast).min().item()
        _logger.info("cache check %s: min cosine real vs cached %.5f", cell, cos)
        assert cos > 0.999, "cached frozen features do not reproduce the model"
    use_features(model, features)
    return model


def embed_encoder(label, cell, seed, model, binder, games) -> Path:
    path = OUT / "embeddings" / f"{label}.db"
    if path.exists():
        table = EmbeddingTable.open(path)
    else:
        metadata = EmbeddingTableMetadata(
            encoder_label=label,
            checkpoint_dir=RUNS / cell / "latest" if seed is None else None,
            embedding_dim=model.embedding_dim,
            binder_versions={game: binder.version_for(game) for game in games},
            created_at=datetime.now(timezone.utc),
        )
        table = EmbeddingTable.create(path, metadata)
    with table:
        added = embed_corpus(model, select_corpus(binder, CorpusSpec(games, None)), table,
                             EMBED_BATCH, "fp16")
    _logger.info("%s: %d cards embedded -> %s", label, added, path)
    return path


def label_sources(holdout) -> list[CardLabels]:
    sources: list[CardLabels] = [GameLabels(), HoldoutTierLabels(holdout)]
    sources += [MetricParquetLabels(name, [METRICS / f"{p}.parquet"]) for name, p in METRIC_LABELS]
    return sources


def analyze(label: str, table_path: Path, sources: list[CardLabels]) -> None:
    with EmbeddingTable.open(table_path) as table:
        run_one(EffectiveRank(), table, OUT / "intrinsic" / label / "unlabeled" / "effective_rank")
        for labels in sources:
            sample = PerLabelCap(PER_LABEL_CAP)
            analyses: list[EmbeddingAnalysis] = [
                ClusterAgreement(labels, sample, seed=SEED),
                LabelCompactness(labels, sample, seed=SEED),
                ProjectionPlot(labels, sample, seed=SEED),
            ]
            for analysis in analyses:
                run_one(analysis, table, OUT / "intrinsic" / label / labels.name / analysis.name)


def run_one(analysis: EmbeddingAnalysis, table: EmbeddingTable, out: Path) -> None:
    shown = out.relative_to(OUT / "intrinsic")
    if out.exists() and any(out.iterdir()):
        _logger.info("skip %s (done)", shown)
        return
    started = time.time()
    try:
        result = analysis.run(table, out)
    except ValueError as error:
        _logger.warning("%s skipped: %s", shown, error)
        return
    scalars = ", ".join(f"{k}={v:.3f}" for k, v in result.scalars.items())
    _logger.info("%s (%.0fs): %s", shown, time.time() - started, scalars)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUT / "intrinsic_suite.log")],
    )
    games = frozenset(g for g in GameId if CardBinder.default_output_path(g).exists())
    binder = CardBinder.load([CardBinder.default_output_path(g) for g in sorted(games)])
    cards = list(select_corpus(binder, CorpusSpec(games, None)))
    _logger.info("corpus: %d cards over %s", len(cards), sorted(g.value for g in games))

    holdout = load_config("single_mixed").plan.holdout
    first = run_training.build_model(load_config("single_mixed").model).to(DEVICE)
    features = frozen_features(first, cards)
    del first
    torch.cuda.empty_cache()

    sources = label_sources(holdout)
    sample = cards[:: max(1, len(cards) // 64)][:64]
    for label, cell, seed in encoders():
        model = build_encoder(cell, seed, features, sample)
        table_path = embed_encoder(label, cell, seed, model, binder, games)
        del model
        torch.cuda.empty_cache()
        analyze(label, table_path, sources)

    curves = {e: EXTRINSIC / e / "rounds.csv" for e in EXTRINSIC_ENCODERS}
    plots = plot_learning_curves(curves, OUT / "extrinsic" / "plots")
    _logger.info("%d learning-curve plots -> %s", len(plots), OUT / "extrinsic" / "plots")


if __name__ == "__main__":
    main()
