"""Evaluate encoder checkpoints, end to end: a first-draft driver showing how
the pieces of src/evaluation/ compose.

For each encoder under test (a trained single-card checkpoint, a trained
multi-card checkpoint, and an untrained single-card baseline):

1. Intrinsic: embed a corpus once into an EmbeddingTable, then run the
   three analyses (cluster agreement, label compactness, projection plot)
   against each label source, plus the label-free ones (effective rank,
   and deck retrieval AUC on VALIDATION decks sampled once per run).
2. Extrinsic: train fresh dojo heads on the frozen encoder (run_extrinsic),
   reusing the same dojo objects for every encoder so each starts from
   identical heads; then overlay every encoder's learning curves.

Everything lands under data/evaluations/<name>/ (see src/evaluation/README.md
for the layout), including run.log, which every invocation appends to. The script is re-runnable: a table that exists is opened
and extended (embed_corpus skips stored cards), and an analysis or
extrinsic run whose output directory already exists is skipped, not redone.

This is a prototype: the corpus (gwent only), label sources, dojos and
hyperparameters are hard-coded below as a worked example; change them in
build_encoders / build_groupings / build_dojos / the spec constants.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/run_evaluation.py --name gwent_v1 \\
        --single-checkpoint runs/single/pretrain_round0007 \\
        --multi-checkpoint runs/multi/pretrain_round0005

    # No checkpoints yet: exercise the whole pipeline on untrained
    # encoders, a small corpus and tiny budgets
    PYTHONPATH=. python3 scripts/run_evaluation.py --name smoke --smoke
"""

import argparse
import itertools
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.gwent_one.faction_mask_metric import FactionMaskMetric
from src.data_refinement.metrics.gwent_one.rarity_mask_metric import RarityMaskMetric
from src.dojos.dojo import Dojo
from src.dojos.gwent_one.masked_field_dojos import FactionMaskDojo, RarityMaskDojo
from src.encoder_model.card_encoder_model import CardEncoderModel
from src.encoder_model.precision import Precision
from src.encoder_model.reference_multicard_models import LinearProjectionMultiCardModel
from src.encoder_model.reference_singlecard_models import LinearProjectionCardModel
from src.evaluation.intrinsic.label_agreement.card_sample import PerLabelCap
from src.evaluation.intrinsic.label_agreement.cluster_agreement import ClusterAgreement
from src.evaluation.intrinsic.deck_retrieval.deck_retrieval_auc import DeckRetrievalAuc
from src.evaluation.intrinsic.effective_rank import EffectiveRank
from src.evaluation.intrinsic.embedding_analysis import EmbeddingAnalysis
from src.evaluation.intrinsic.label_agreement.label_compactness import LabelCompactness
from src.evaluation.intrinsic.projection.projection_plot import ProjectionPlot
from src.evaluation.intrinsic.deck_retrieval.evaluation_decks import (
    EvaluationDeck,
    EvaluationDeckSpec,
    sample_evaluation_decks,
)
from src.evaluation.intrinsic.embedding.embed_corpus import embed_corpus
from src.evaluation.intrinsic.embedding.embedding_table import EmbeddingTable
from src.evaluation.intrinsic.embedding.embedding_table_metadata import EmbeddingTableMetadata
from src.evaluation.extrinsic.learning_curves import plot_learning_curves
from src.evaluation.extrinsic.run_extrinsic import ExtrinsicSpec, run_extrinsic
from src.evaluation.intrinsic.cluster_labels.cluster_labels import ClusterLabels, HoldoutTierClusterLabels
from src.evaluation.intrinsic.cluster_labels.metric_parquet_cluster_labels import MetricParquetClusterLabels
from src.evaluation.intrinsic.embedding.select_corpus import CorpusSpec, select_corpus
from src.run_logging import (
    RUN_LOG_NAME,
    add_logging_arguments,
    configure_logging,
    logging_settings_from,
    script_logger,
)
from src.schema.card import GenericCard
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec, UnknownDeckSplitError
from src.training.plan import HardwareLimits, SaturationSpec, Uniform
from src.training.recording.checkpointer import (
    load_checkpoint_holdout,
    load_encoder_weights,
)

_logger = script_logger(__file__)

_BINDER_PATH = Path("data/final/cards/gwent.jsonl")
_OUTPUT_ROOT = Path("data/evaluations")
_GAMES = frozenset({GameId.GWENT})
# Every encoder and every dojo head must agree on this width
_EMBED_DIM = 256
_SEED = 0


@dataclass(frozen=True)
class EncoderUnderTest:
    """One encoder to evaluate. checkpoint_dir is None for an untrained
    baseline (recorded in its embedding table's metadata)."""

    label: str
    model: CardEncoderModel
    checkpoint_dir: Path | None


@dataclass(frozen=True)
class Budget:
    """How much work each stage does: full size, or tiny for --smoke."""

    max_cards: int | None  # None = the whole corpus
    per_label_cap: int
    deck_count: int  # VALIDATION decks per game, for deck retrieval AUC
    embed_batch_size: int
    extrinsic: ExtrinsicSpec


def build_encoders(args: argparse.Namespace) -> list[EncoderUnderTest]:
    """The encoders to compare. A checkpoint's weights are loaded into a
    freshly built model of the architecture it was trained as (the
    checkpoint does not record its architecture yet - see the plan's open
    questions). Every model is moved to args.device after its weights load;
    extrinsic dojo heads follow it there (Trainer.run moves them)."""
    result: list[EncoderUnderTest] = []
    if args.single_checkpoint:
        model = LinearProjectionCardModel(card_embedding_size=_EMBED_DIM)
        load_encoder_weights(args.single_checkpoint, model)
        model.to(args.device)
        result.append(EncoderUnderTest("single", model, args.single_checkpoint))
    if args.multi_checkpoint:
        multi = LinearProjectionMultiCardModel(card_embedding_size=_EMBED_DIM)
        load_encoder_weights(args.multi_checkpoint, multi)
        multi.to(args.device)
        result.append(EncoderUnderTest("multi", multi, args.multi_checkpoint))
    # The untrained baseline: same architecture, pretrained text encoder,
    # randomly initialised head - what training has to beat
    untrained = LinearProjectionCardModel(card_embedding_size=_EMBED_DIM)
    untrained.to(args.device)
    result.append(EncoderUnderTest("untrained", untrained, None))
    return result


def training_holdout(args: argparse.Namespace) -> HoldoutSpec:
    """The HoldoutSpec the trained encoders used, from the first checkpoint
    given; no holdout if none was. Used for holdout-tier labels and to build
    the extrinsic dojos, so tiers mean the same thing everywhere."""
    for checkpoint in (args.single_checkpoint, args.multi_checkpoint):
        if checkpoint:
            return load_checkpoint_holdout(checkpoint)
    return HoldoutSpec.no_holdout(seed=_SEED)


def build_groupings(holdout: HoldoutSpec) -> list[ClusterLabels]:
    """What to color, cluster and score the embeddings by. A gwent-only
    corpus has one game, so GameClusterLabels would be refused (one label);
    within-game labels come from the gwent metrics' parquets."""
    return [
        MetricParquetClusterLabels("faction", [FactionMaskMetric.DEFAULT_OUTPUT_PATH]),
        MetricParquetClusterLabels("rarity", [RarityMaskMetric.DEFAULT_OUTPUT_PATH]),
        # Seen in training vs. held out: the domain-shift view
        HoldoutTierClusterLabels(holdout),
    ]


def build_dojos(binder: CardBinder, holdout: HoldoutSpec) -> list[Dojo]:
    """The extrinsic tasks. Built once and reused for every encoder, so
    each run starts from the same as-built heads (run_extrinsic resets
    them). Masked-field dojos still mask their target field."""
    return [
        FactionMaskDojo(binder, holdout, _EMBED_DIM, rng_seed=_SEED),
        RarityMaskDojo(binder, holdout, _EMBED_DIM, rng_seed=_SEED),
    ]


def build_deck_auc(holdout: HoldoutSpec, budget: Budget) -> DeckRetrievalAuc | None:
    """Deck retrieval AUC over each corpus game's VALIDATION decks, sampled
    once so every encoder is scored on the same decks. None (logged) when
    no game has a deck box or a qualifying deck, or the holdout has no deck
    split (an old checkpoint): the analysis is skipped, not the run. A game
    whose deck box cannot be read (sqlite3.Error) is logged and skipped."""
    decks: list[EvaluationDeck] = []
    spec = EvaluationDeckSpec(count=budget.deck_count, seed=_SEED)

    # Sample each game with a deck box
    for game in sorted(_GAMES):
        path = DeckBox.default_output_path(game)
        if not path.exists():
            _logger.warning("deck retrieval AUC: no deck box for %s at %s", game.value, path)
            continue
        try:
            decks.extend(_sample_game_decks(path, game, holdout, spec))
        except UnknownDeckSplitError as error:
            _logger.warning("deck retrieval AUC skipped: %s", error)
            return None
        except sqlite3.Error as error:
            _logger.warning("deck retrieval AUC: %s unreadable: %s", path, error)

    # Nothing to score is a skip, not an error
    if not decks:
        _logger.warning("deck retrieval AUC skipped: no evaluation decks")
        return None
    _logger.info("      deck retrieval AUC: %d VALIDATION decks", len(decks))
    return DeckRetrievalAuc(decks, holdout, seed=_SEED)


def _sample_game_decks(
    path: Path, game: GameId, holdout: HoldoutSpec, spec: EvaluationDeckSpec
) -> tuple[EvaluationDeck, ...]:
    """Open one game's deck box (it exists, so DeckBox.load creates nothing)
    and sample it, reading only. The connection is left to garbage
    collection (DeckBox has no close). Raises UnknownDeckSplitError (the
    caller skips the analysis) and sqlite3.Error (the caller skips the
    game)."""
    deck_box = DeckBox.load([path])
    return sample_evaluation_decks(deck_box, game, holdout, spec)


def budget_for(smoke: bool) -> Budget:
    """Full-size settings, or tiny ones that only prove the wiring."""
    if smoke:
        spec = ExtrinsicSpec(
            diet_rule=Uniform(),
            head_lr=1e-3,
            steps_per_round=5,
            max_rounds=3,
            saturation=_saturation(patience_rounds=10),  # never saturates early
            eval_examples_per_dojo=32,
            seed=_SEED,
        )
        return Budget(
            max_cards=200,
            per_label_cap=20,
            deck_count=50,
            embed_batch_size=16,
            extrinsic=spec,
        )
    spec = ExtrinsicSpec(
        diet_rule=Uniform(),
        head_lr=1e-3,
        # Short rounds: a head on ~1k TRAIN cards overfits within a few
        # hundred steps, and it trains patience_rounds rounds past its best
        steps_per_round=50,
        max_rounds=60,
        saturation=_saturation(patience_rounds=3),
        eval_examples_per_dojo=512,
        seed=_SEED,
    )
    return Budget(
        max_cards=None,
        per_label_cap=500,
        deck_count=1000,
        embed_batch_size=64,
        extrinsic=spec,
    )


def _saturation(patience_rounds: int) -> SaturationSpec:
    """Saturation for an extrinsic run. Never reactivates: the encoder is
    frozen and each dojo has its own head, so a rising TEST loss can only be
    that head overfitting, and training it more would make it worse."""
    return SaturationSpec(
        epsilon=1e-3,
        patience_rounds=patience_rounds,
        reactivation_delta=math.inf,
        target_saturated_fraction=1.0,
    )


def embed_encoder(
    encoder: EncoderUnderTest,
    binder: CardBinder,
    run_dir: Path,
    budget: Budget,
    precision: Precision,
) -> Path:
    """Stage 1a: embed the corpus into this encoder's table (created, or
    opened and extended if it already exists). Returns the table's path."""
    path = run_dir / "embeddings" / f"{encoder.label}.db"
    corpus: Iterator[GenericCard] = select_corpus(binder, CorpusSpec(_GAMES, None))
    if budget.max_cards is not None:
        corpus = itertools.islice(corpus, budget.max_cards)

    if path.exists():
        table = EmbeddingTable.open(path)
    else:
        metadata = EmbeddingTableMetadata(
            encoder_label=encoder.label,
            checkpoint_dir=encoder.checkpoint_dir,
            embedding_dim=encoder.model.embedding_dim,
            binder_versions={game: binder.version_for(game) for game in _GAMES},
            created_at=datetime.now(timezone.utc),
        )
        table = EmbeddingTable.create(path, metadata)
    with table:
        added = embed_corpus(
            encoder.model, corpus, table, budget.embed_batch_size, precision
        )
    _logger.info("      %s: %d cards embedded -> %s", encoder.label, added, path)
    return path


def analyze_encoder(
    encoder: EncoderUnderTest,
    table_path: Path,
    groupings: list[ClusterLabels],
    deck_auc: DeckRetrievalAuc | None,
    run_dir: Path,
    budget: Budget,
) -> None:
    """Stage 1b: every analysis x every label source, into
    intrinsic/<encoder>/<label source>/<analysis>/, plus the label-free
    analyses (deck_auc when not None) into
    intrinsic/<encoder>/unlabeled/<analysis>/. Existing results are kept
    (skipped), so an interrupted run can be resumed."""
    with EmbeddingTable.open(table_path) as table:
        unlabeled = run_dir / "intrinsic" / encoder.label / "unlabeled"
        label_free: list[EmbeddingAnalysis] = [EffectiveRank()]
        if deck_auc is not None:
            label_free.append(deck_auc)
        for analysis in label_free:
            _run_analysis(analysis, table, unlabeled / analysis.name)
        for cluster_labels in groupings:
            sample = PerLabelCap(budget.per_label_cap)
            analyses: list[EmbeddingAnalysis] = [
                ClusterAgreement(cluster_labels, sample, seed=_SEED),
                LabelCompactness(cluster_labels, sample, seed=_SEED),
                ProjectionPlot(cluster_labels, sample, seed=_SEED),
            ]
            for analysis in analyses:
                out = (
                    run_dir / "intrinsic" / encoder.label / cluster_labels.grouping / analysis.name
                )
                _run_analysis(analysis, table, out)


def _run_analysis(
    analysis: EmbeddingAnalysis, table: EmbeddingTable, out: Path
) -> None:
    """One analysis, logged and skipped on failure - a label source that
    doesn't fit this corpus (e.g. one label) must not stop the others."""
    # Log paths from intrinsic/ down: <encoder>/<grouping>/<analysis>
    shown = out.relative_to(out.parents[3])
    if out.exists() and any(out.iterdir()):
        _logger.info("      skip %s (already done)", shown)
        return
    try:
        result = analysis.run(table, out)
    except ValueError as error:
        _logger.warning("%s skipped: %s", out, error)
        return
    scalars = ", ".join(f"{key}={value:.3f}" for key, value in result.scalars.items())
    _logger.info("      %s: %s", shown, scalars)


def run_extrinsic_all(
    encoders: list[EncoderUnderTest],
    dojos: list[Dojo],
    run_dir: Path,
    budget: Budget,
    hardware: HardwareLimits,
) -> dict[str, Path]:
    """Stage 2: one extrinsic run per encoder on the same dojos. Returns
    encoder label -> rounds CSV for plotting (existing runs are reused)."""
    result: dict[str, Path] = {}
    for encoder in encoders:
        rounds_csv = run_dir / "extrinsic" / encoder.label / "rounds.csv"
        if not rounds_csv.parent.exists():
            outcome = run_extrinsic(
                encoder.model, encoder.label, dojos, budget.extrinsic, hardware, run_dir
            )
            normalized = {
                name: round(split_loss.normalized, 4)
                for name, split_loss in (outcome.validation_losses or {}).items()
            }
            _logger.info(
                "      %s: validation (x baseline)=%s stopped_early=%s",
                encoder.label,
                normalized,
                outcome.stopped_early_reason,
            )
        else:
            _logger.info("      skip extrinsic/%s (already done)", encoder.label)
        if rounds_csv.exists():
            result[encoder.label] = rounds_csv
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="data/evaluations/<name>/")
    parser.add_argument("--single-checkpoint", type=Path)
    parser.add_argument("--multi-checkpoint", type=Path)
    parser.add_argument("--binder-path", type=Path, default=_BINDER_PATH)
    parser.add_argument("--output-root", type=Path, default=_OUTPUT_ROOT)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--max-batch-cost", type=int, default=64)
    parser.add_argument(
        "--device", type=torch.device, default=torch.device("cpu"), help="e.g. cuda"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="tiny corpus and budgets: wiring check"
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.name
    configure_logging(logging_settings_from(args, run_dir / RUN_LOG_NAME))
    budget = budget_for(args.smoke)
    hardware = HardwareLimits(args.max_batch_cost, args.precision)

    _logger.info("[1/5] Loading the card binder from %s", args.binder_path)
    binder = CardBinder.load([args.binder_path])

    _logger.info("[2/5] Building encoders, label sources and dojos")
    holdout = training_holdout(args)
    encoders = build_encoders(args)
    groupings = build_groupings(holdout)
    dojos = build_dojos(binder, holdout)
    deck_auc = build_deck_auc(holdout, budget)
    _logger.info("      encoders: %s", [encoder.label for encoder in encoders])

    _logger.info("[3/5] Intrinsic: embedding each encoder's corpus")
    tables = {
        encoder.label: embed_encoder(encoder, binder, run_dir, budget, args.precision)
        for encoder in encoders
    }

    _logger.info("[4/5] Intrinsic: analyses")
    for encoder in encoders:
        analyze_encoder(
            encoder, tables[encoder.label], groupings, deck_auc, run_dir, budget
        )

    _logger.info(
        "[5/5] Extrinsic: fresh heads on each frozen encoder, then learning curves"
    )
    curves = run_extrinsic_all(encoders, dojos, run_dir, budget, hardware)
    if curves:
        plots = plot_learning_curves(curves, run_dir / "extrinsic" / "plots")
        _logger.info(
            "      %d learning-curve plots -> %s",
            len(plots),
            run_dir / "extrinsic" / "plots",
        )

    _logger.info("Done: %s", run_dir)


if __name__ == "__main__":
    main()
