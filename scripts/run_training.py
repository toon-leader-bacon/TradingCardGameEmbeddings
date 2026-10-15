"""Train a card-embedding encoder from a run config file.

Reads a YAML config (see configs/training/), applies any --set overrides,
builds the model and the named dojos, checks every dojo once (preflight),
then calls Trainer.run(). Everything a run produces lands in its
run_directory, which must not exist yet:

    <run_directory>/
        run_config.yaml     the config as run (overrides applied)
        rounds.csv          one row per (round, dojo)
        checkpoints.csv     one row per best checkpoint written (only the
                            last best per phase is still on disk)
        diet_shares.csv     each phase's expected share of steps per dojo
        <phase>_roundNNNN/  the phase's best checkpoint (state.pt, encoder.pt, ...)
        latest/             the most recent round, rewritten every round
        run.log             the full log, from the script's first line

The log also goes to logs/run_training/<timestamp>.log (or --log-file),
which is the only log for a --check.

Usage (from the project root, in the ROCm venv for a GPU run):

    PYTHONPATH=. python scripts/run_training.py configs/training/gpu_smoke.yaml
    PYTHONPATH=. python scripts/run_training.py configs/training/gpu_smoke.yaml \\
        --set run_directory=data/runs/smoke_2 --set phases.0.max_rounds=2

    # Build and preflight the dojos, build the model (may download it) and
    # validate the plan against both; write nothing, train nothing
    PYTHONPATH=. python scripts/run_training.py configs/training/gpu_smoke.yaml --check
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import yaml

from src.dojos.dojo import BatchBudget, Dojo
from src.dojos.loss.regression_objective import RegressionObjective
from src.dojos.mods.mod import ModTally
from src.encoder_model.card_embedding_pipeline import CardEmbeddingPipeline
from src.encoder_model.card_encoder_model import CardEncoderModel
from src.encoder_model.embedding_head import (
    EmbeddingHead,
    LinearEmbeddingHead,
    ResidualMlpEmbeddingHead,
)
from src.encoder_model.multi_card_model import MultiCardModel
from src.encoder_model.pooling.attention_pooling import AttentionPooling
from src.encoder_model.pooling.mean_pooling import MeanPooling
from src.encoder_model.pooling.token_pooling import TokenPooling
from src.encoder_model.pooling.transformer_pooling import TransformerPooling
from src.encoder_model.enrichment.token_whitening import TokenWhitening
from src.encoder_model.whitening.whitening_stats import WhiteningStats
from src.encoder_model.single_card_model import SingleCardModel
from src.encoder_model.text_encoder import PretrainedTextEncoder, TextEncoder
from src.run_logging import (
    RUN_LOG_NAME,
    LoggingSettings,
    add_logging_arguments,
    configure_logging,
    logging_settings_from,
    mirror_log_into,
    script_log_path,
    script_logger,
)
from src.schema.card import GenericCard
from src.schema.holdout import HoldoutSpec
from src.training.diet.diet_shares import (
    format_diet_shares,
    plan_diet_shares,
    write_diet_shares_csv,
)
from src.training.dojo_catalog import DOJO_CATALOG, build_dojos
from src.training.dojo_recipes.card_shelf import CardShelf
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext
from src.training.preflight import preflight_dojo
from src.training.recording.checkpointer import DirectoryCheckpointer
from src.training.recording.reports import TrainingResult
from src.training.recording.run_listener import (
    CsvRunListener,
    LoggingRunListener,
    RunListener,
)
from src.training.model_spec import (
    AttentionPoolingSpec,
    EmbeddingHeadKind,
    MeanPoolingSpec,
    ModelSpec,
    PoolingSpec,
    TokenWhiteningSpec,
    TransformerPoolingSpec,
)
from src.training.run_config import (
    ConfigDocument,
    RunConfig,
    apply_overrides,
    parse_run_config,
    read_config_document,
)
from src.training.trainer import Trainer

_CONFIG_COPY_NAME = "run_config.yaml"
_WHITENING_STATS_COPY_NAME = "whitening_stats.pt"
_ROUNDS_CSV_NAME = "rounds.csv"
_CHECKPOINTS_CSV_NAME = "checkpoints.csv"
_DIET_SHARES_CSV_NAME = "diet_shares.csv"

_logger = script_logger(__file__)


def card_cost(card: GenericCard) -> int:
    """One card's share of HardwareLimits.max_batch_cost: 1, so the budget
    counts cards. Preflight and training both batch by it.

    Inputs: card. Output: 1. Side effects: none. Exceptions: none.

    Example:
        >>> card_cost(card)
        1
    """
    return 1


@dataclass(frozen=True)
class CommandLine:
    """The parsed command line.

    config: the YAML run config. overrides: "dotted.key=value" strings, in
    order. check_only: build and preflight the dojos, build the model and
    validate the plan with a Trainer, then stop without training.
    logging: where and how verbosely to log.
    """

    config: Path
    overrides: tuple[str, ...]
    check_only: bool
    logging: LoggingSettings


def parse_command_line() -> CommandLine:
    """Parse sys.argv into a CommandLine.

    Inputs: none (reads sys.argv). Output: CommandLine.
    Side effects: argparse prints usage and exits on bad arguments.
    Exceptions: SystemExit from argparse.

    Example:
        >>> parse_command_line().config
        PosixPath('configs/training/gpu_smoke.yaml')
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path, help="YAML run config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override one config value (repeatable; later wins)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build and preflight the dojos, build the model and validate "
        "the plan, then stop without training",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    return CommandLine(
        config=args.config,
        overrides=tuple(args.overrides),
        check_only=args.check,
        logging=logging_settings_from(args, script_log_path(__file__)),
    )


def main() -> int:
    """Run one training run end to end; return a process exit code.

    Output: 0 on a completed run (even one that stopped early: the reason
        is logged) or a passed --check (preflight, model build and plan
        validation), 1 if preflight failed.
    Side effects: loads card data, may write split files, creates the run
        directory and everything in it, trains on the configured device,
        logs progress.
    Exceptions: ValueError for a bad config (including an unknown dojo)
        or an existing run directory; RuntimeError if the configured
        device is unavailable; FileNotFoundError for missing card data,
        and whatever else dojo construction raises (a stale metric).

    Example:
        >>> raise SystemExit(main())
    """
    command_line = parse_command_line()
    configure_logging(command_line.logging)

    # Read, override and parse the config; fail fast before loading anything
    document = apply_overrides(
        read_config_document(command_line.config), command_line.overrides
    )
    config = parse_run_config(document, tuple(DOJO_CATALOG))
    _require_device(config.device)
    require_whitening_fits_run(config.model, config.plan.holdout)
    if not command_line.check_only:
        _require_new_run_directory(config.run_directory)

    # Build the dojos and check each one before committing to a run
    context = DojoBuildContext(
        shelf=CardShelf(),
        holdout=config.plan.holdout,
        card_embedding_size=config.model.card_embedding_size,
        rng_seed=config.plan.seed,
        mod_overrides=config.mod_overrides,
        staple_thresholds=config.staple_thresholds,
        regression_objective=RegressionObjective.for_kind(config.regression_loss),
    )
    dojos = build_dojos(config.dojo_names, context)
    if not preflight_passes(dojos, config):
        return 1
    # Each phase's expected diet, checked by eye before any GPU time
    shares = plan_diet_shares(config.plan, dojos)
    for line in format_diet_shares(shares):
        _logger.info("%s", line)
    model = build_model(config.model).to(config.device)
    _logger.info(
        "Model: %s trainable parameters", f"{trainable_parameter_count(model):,}"
    )
    if command_line.check_only:
        # Constructing a Trainer validates the plan against the model and
        # dojos (e.g. a phase with nothing trainable) without training
        Trainer(model, dojos, config.plan, config.limits, None, [], cost_of=card_cost)
        _logger.info("Check passed; nothing trained.")
        return 0

    # Train, recording into a fresh run directory
    recorded = _with_run_whitening_stats(_with_expanded_dojos(document, config), config)
    _create_run_directory(config.run_directory, recorded)
    mirror_log_into(config.run_directory / RUN_LOG_NAME)
    _copy_whitening_stats(config)
    write_diet_shares_csv(shares, config.run_directory / _DIET_SHARES_CSV_NAME)
    result = train(model, dojos, config)
    log_summary(result, config.run_directory)
    return 0


def build_model(spec: ModelSpec) -> CardEncoderModel:
    """Construct the model spec describes (on the CPU): a CardEmbeddingPipeline
    (frozen pretrained text encoder, pooling, embedding head) as a
    SingleCardModel, or as a MultiCardModel when spec has group attention.

    Inputs: spec (ModelSpec). Output: CardEncoderModel.
    Side effects: may download the pretrained checkpoint.
    Exceptions: whatever the text encoder or model constructors raise.

    Example:
        >>> build_model(ModelSpec(EmbeddingHeadKind.RESIDUAL_MLP, "answerdotai/ModernBERT-base", 256))
    """
    text_encoder = PretrainedTextEncoder(checkpoint=spec.checkpoint, trainable=False)
    card_pipeline = build_card_pipeline(spec, text_encoder)
    attention = spec.group_attention
    if attention is None:
        return SingleCardModel(card_pipeline)
    return MultiCardModel(
        card_pipeline,
        num_heads=attention.num_heads,
        num_layers=attention.num_layers,
        ffn_dim=attention.ffn_dim,
        dropout=attention.dropout,
        norm_first=attention.norm_first,
    )


def build_card_pipeline(
    spec: ModelSpec, text_encoder: TextEncoder
) -> CardEmbeddingPipeline:
    """The CardEmbeddingPipeline spec describes, around the given text
    encoder: spec.enrichment (if any), spec.pooling, then the head
    spec.embedding_head names, each built from the previous stage's
    output_dim, ending at spec.card_embedding_size.

    Inputs: spec, text_encoder (built by the caller, so tests can inject a
        fake). Output: CardEmbeddingPipeline.
    Side effects: reads the whitening stats file when spec has token
        whitening.
    Exceptions: ValueError if the whitening stats were fitted on another
        checkpoint than spec.checkpoint, or a stage rejects its width
        (e.g. transformer heads that do not divide it); FileNotFoundError
        for a missing stats file.

    Example:
        >>> build_card_pipeline(spec, text_encoder).output_dim
        256
    """
    # Each stage reads the width the previous one produces
    enrichment = None
    token_width = text_encoder.output_dim
    if spec.enrichment is not None:
        enrichment = _build_token_whitening(spec.enrichment, spec.checkpoint)
        token_width = enrichment.output_dim
    pooling = _build_pooling(spec.pooling, token_width)
    head = _build_head(spec, pooling.output_dim)
    return CardEmbeddingPipeline(text_encoder, enrichment, pooling, head)


def _build_token_whitening(spec: TokenWhiteningSpec, checkpoint: str) -> TokenWhitening:
    """Private helper - single consumer is build_card_pipeline.

    Inputs: spec, checkpoint (the run's text encoder).
    Output: TokenWhitening from the stats at spec.stats_path.
    Side effects: reads the stats file.
    Exceptions: FileNotFoundError; ValueError if the stats were fitted on
        another checkpoint.
    """
    stats = WhiteningStats.load(spec.stats_path)
    stats.require_fitted_on(checkpoint)
    return TokenWhitening(stats)


def _build_pooling(spec: PoolingSpec, token_width: int) -> TokenPooling:
    """Private helper - single consumer is build_card_pipeline. A
    transformer's inner pooling is built by _build_simple_pooling, so
    there is no recursion (the spec type forbids nesting).

    Inputs: spec, token_width. Output: the TokenPooling spec names.
    Side effects: none. Exceptions: ValueError from the pooling.
    """
    if isinstance(spec, TransformerPoolingSpec):
        return TransformerPooling(
            token_width,
            inner=_build_simple_pooling(spec.inner, token_width),
            num_layers=spec.num_layers,
            num_heads=spec.num_heads,
            ffn_dim=spec.ffn_dim,
            dropout=spec.dropout,
        )
    return _build_simple_pooling(spec, token_width)


def _build_simple_pooling(
    spec: MeanPoolingSpec | AttentionPoolingSpec, token_width: int
) -> MeanPooling | AttentionPooling:
    """Private helper - consumers are _build_pooling (a top-level pooling
    and a transformer's inner one).

    Inputs: spec, token_width. Output: MeanPooling or AttentionPooling.
    Side effects: none. Exceptions: ValueError from the pooling.
    """
    if isinstance(spec, AttentionPoolingSpec):
        return AttentionPooling(token_width, num_queries=spec.num_queries)
    return MeanPooling(token_width)


def _build_head(spec: ModelSpec, pooled_width: int) -> EmbeddingHead:
    """Private helper - single consumer is build_card_pipeline.

    Inputs: spec (embedding_head, card_embedding_size, mlp_*),
        pooled_width. Output: the EmbeddingHead spec.embedding_head names.
    Side effects: none. Exceptions: none.
    """
    match spec.embedding_head:
        case EmbeddingHeadKind.LINEAR_PROJECTION:
            return LinearEmbeddingHead(pooled_width, spec.card_embedding_size)
        case EmbeddingHeadKind.RESIDUAL_MLP:
            return ResidualMlpEmbeddingHead(
                pooled_width,
                spec.card_embedding_size,
                hidden_dim=spec.mlp_hidden_dim,
                num_blocks=spec.mlp_num_blocks,
            )


def require_whitening_fits_run(spec: ModelSpec, holdout: HoldoutSpec) -> None:
    """Refuse token-whitening stats the run cannot use, before any dojo is
    built: fitted on another text encoder than spec.checkpoint, or under
    another holdout than the run's (their sample may include this run's
    TEST or VALIDATION cards). No-op without token whitening.

    Inputs: spec (the run's model), holdout (the run's).
    Output: none.
    Side effects: reads the stats file when spec has token whitening.
    Exceptions: ValueError on either mismatch; FileNotFoundError for a
        missing stats file.

    Example:
        >>> require_whitening_fits_run(config.model, config.plan.holdout)
    """
    if spec.enrichment is None:
        return
    stats = WhiteningStats.load(spec.enrichment.stats_path)
    stats.require_fitted_on(spec.checkpoint)
    if stats.holdout != holdout:
        raise ValueError(
            f"{spec.enrichment.stats_path} was fitted under holdout "
            f"{stats.holdout}, but this run uses {holdout}; refit it with "
            "scripts/fit_whitening.py --holdout-from <this config>"
        )


def _with_run_whitening_stats(
    document: ConfigDocument, config: RunConfig
) -> ConfigDocument:
    """document with model.enrichment.stats_path pointing at the run
    directory's own copy (_WHITENING_STATS_COPY_NAME), when the run has
    token whitening; otherwise document itself. So a run directory stays
    self-contained (see _copy_whitening_stats).

    Inputs: document (not modified), config (parsed from it).
    Output: a copy of document with the one path replaced, or document.
    Side effects: none. Exceptions: none.
    """
    if config.model.enrichment is None:
        return document
    copy_path = config.run_directory / _WHITENING_STATS_COPY_NAME
    model = dict(document["model"])
    model["enrichment"] = {**model["enrichment"], "stats_path": copy_path.as_posix()}
    return {**document, "model": model}


def _copy_whitening_stats(config: RunConfig) -> None:
    """Copy the run's whitening stats file into its run directory, when the
    run has token whitening.

    Inputs: config. Output: none.
    Side effects: writes one file into config.run_directory.
    Exceptions: OSError from the filesystem.
    """
    if config.model.enrichment is not None:
        shutil.copyfile(
            config.model.enrichment.stats_path,
            config.run_directory / _WHITENING_STATS_COPY_NAME,
        )


def trainable_parameter_count(model: torch.nn.Module) -> int:
    """How many parameters training updates (the frozen text encoder's are
    not counted), for comparing model sizes across runs.

    Inputs: model. Output: int. Side effects: none. Exceptions: none.

    Example:
        >>> trainable_parameter_count(torch.nn.Linear(2, 3))
        9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def preflight_passes(dojos: list[Dojo], config: RunConfig) -> bool:
    """Run preflight_dojo on every dojo and log one line each.

    Inputs: dojos (built dojos), config (for the batch budget and card_embedding_size).
    Output: True if every dojo passed.
    Side effects: logs a report; reads one TRAIN batch per dojo.
    Exceptions: none (a failing dojo is reported, not raised).

    Example:
        >>> preflight_passes(dojos, config)
        True
    """
    budget = BatchBudget(config.limits.max_batch_cost, card_cost)
    _logger.info("Preflight: %d dojos", len(dojos))
    passed = 0
    for dojo in dojos:
        check = preflight_dojo(dojo, budget, config.model.card_embedding_size)
        if check.ok:
            passed += 1
            _logger.info(
                "[ OK ] %s: train=%d test=%d sample loss=%.4g baseline=%.4g",
                check.dojo_name,
                check.train_count,
                check.test_count,
                check.sample_loss,
                check.sample_baseline_loss,
            )
        else:
            _logger.error("[FAIL] %s: %s", check.dojo_name, check.error)
        log_mod_tallies(check.mod_tallies)
    _logger.info("%d/%d dojos passed preflight", passed, len(dojos))
    return passed == len(dojos)


def log_mod_tallies(tallies: Mapping[str, ModTally]) -> None:
    """One indented line per augmentation mod: cards seen, changed and
    failed over the preflight batch, with a WARNING on a mod that changed
    nothing (e.g. its keys no longer match the data) or failed on any card
    (with its first error). Informational: never fails preflight.

    Inputs: tallies (a PreflightResult's snapshot). Output: none.
    Side effects: logs one line per mod, at WARNING for a suspect mod.
        Exceptions: none.

    Example:
        >>> log_mod_tallies(check.mod_tallies)
               0:ShuffleKeysMod: changed 30/32, failed 0
    """
    for label, tally in tallies.items():
        line = (
            f"       {label}: changed {tally.cards_changed}/{tally.cards_seen}, "
            f"failed {tally.cards_failed}"
        )
        if tally.cards_failed:
            _logger.warning("%s  first failure %s", line, tally.first_failure)
        elif tally.cards_seen and not tally.cards_changed:
            _logger.warning(
                "%s  changed no card (do its fields still match the data?)", line
            )
        else:
            _logger.info("%s", line)


def train(
    model: CardEncoderModel, dojos: list[Dojo], config: RunConfig
) -> TrainingResult:
    """Construct the Trainer with a checkpointer and the logging and CSV
    listeners pointed at config.run_directory, and run it.

    Inputs: model (already on config.device), dojos, config.
    Output: TrainingResult.
    Side effects: trains model and dojo heads; writes checkpoints and CSVs.
    Exceptions: ValueError from the Trainer constructor for a plan/dojo
        mismatch; KeyboardInterrupt propagates.

    Example:
        >>> result = train(model, dojos, config)
    """
    run_directory = config.run_directory
    listeners: list[RunListener] = [
        LoggingRunListener(),
        CsvRunListener(
            run_directory / _ROUNDS_CSV_NAME, run_directory / _CHECKPOINTS_CSV_NAME
        ),
    ]
    trainer = Trainer(
        model=model,
        dojos=dojos,
        plan=config.plan,
        limits=config.limits,
        checkpointer=DirectoryCheckpointer(run_directory),
        listeners=listeners,
        cost_of=card_cost,
    )
    return trainer.run()


def log_summary(result: TrainingResult, run_directory: Path) -> None:
    """Log why the run ended, its final per-dojo TEST loss and where the
    best checkpoint and logs are.

    Inputs: result (TrainingResult), run_directory (Path). Output: none.
    Side effects: logs. Exceptions: none.

    Example:
        >>> log_summary(train(model, dojos, config), config.run_directory)
    """
    _logger.info("stopped early: %s", result.stopped_early_reason or "no")
    if result.final_report is not None:
        _logger.info("final round %d TEST loss:", result.final_report.round_index)
        for name, split_loss in sorted(result.final_report.per_dojo_test_loss.items()):
            _logger.info(
                "  %s: %.4g (%.3fx baseline)",
                name,
                split_loss.loss,
                split_loss.normalized,
            )
    if result.best_checkpoint is not None:
        _logger.info("best checkpoint: %s", result.best_checkpoint.path)
    _logger.info("rounds: %s", run_directory / _ROUNDS_CSV_NAME)
    _logger.info("log: %s", run_directory / RUN_LOG_NAME)


def _require_device(device: torch.device) -> None:
    """Fail before loading anything if device cannot be used.

    Inputs: device. Output: none. Side effects: none.
    Exceptions: RuntimeError if device is a CUDA device and
        torch.cuda.is_available() is False (e.g. the CPU-only venv).
    """
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {device} requested but torch {torch.__version__} sees no "
            "GPU; use the ROCm venv or set device: cpu"
        )


def _require_new_run_directory(run_directory: Path) -> None:
    """Refuse to reuse a run directory, so a run never mixes its CSVs or
    checkpoints with another's.

    Inputs: run_directory. Output: none. Side effects: none.
    Exceptions: ValueError if run_directory already exists.
    """
    if run_directory.exists():
        raise ValueError(
            f"run directory {run_directory} already exists; pick a new "
            "run_directory (e.g. --set run_directory=data/runs/<name>)"
        )


def _with_expanded_dojos(document: ConfigDocument, config: RunConfig) -> ConfigDocument:
    """document with "dojos" and "held_out_dojos" replaced by the names
    their patterns expanded to, so the run's config copy still names the
    same dojos after the catalog grows.

    Inputs: document (not modified), config (parsed from it).
    Output: a shallow copy of document. Side effects: none. Exceptions: none.
    """
    held_out = config.plan.held_out_dojos
    result = dict(document)
    result["dojos"] = list(config.dojo_names)
    result["held_out_dojos"] = [name for name in config.dojo_names if name in held_out]
    return result


def _create_run_directory(run_directory: Path, document: ConfigDocument) -> None:
    """Create run_directory and write document into it as run_config.yaml.

    Inputs: run_directory, document (the config as run). Output: none.
    Side effects: creates directories and one file.
    Exceptions: OSError from the filesystem.
    """
    run_directory.mkdir(parents=True)
    with (run_directory / _CONFIG_COPY_NAME).open("w", encoding="utf-8") as copy_file:
        yaml.safe_dump(document, copy_file, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
