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
        <phase>_roundNNNN/  the phase's best checkpoint (state.pt, encoder.pt, ...)
        latest/             the most recent round, rewritten every round

Usage (from the project root, in the ROCm venv for a GPU run):

    PYTHONPATH=. python scripts/run_training.py configs/training/gpu_smoke.yaml
    PYTHONPATH=. python scripts/run_training.py configs/training/gpu_smoke.yaml \\
        --set run_directory=data/runs/smoke_2 --set phases.0.max_rounds=2

    # Build and preflight the dojos, build the model (may download it) and
    # validate the plan against both; write nothing, train nothing
    PYTHONPATH=. python scripts/run_training.py configs/training/gpu_smoke.yaml --check
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import torch
import yaml

from src.dojos.dojo import BatchBudget, Dojo
from src.dojos.loss.regression_objective import RegressionObjective
from src.dojos.mods.mod import ModTally
from src.encoder_model.card_encoder_model import CardEncoderModel
from src.encoder_model.reference_singlecard_models import (
    AttentionPoolingCardModel,
    LinearProjectionCardModel,
    ResidualMlpCardModel,
)
from src.schema.card import GenericCard
from src.training.dojo_catalog import CardShelf, DojoBuildContext, build_dojos
from src.training.preflight import preflight_dojo
from src.training.recording.checkpointer import DirectoryCheckpointer
from src.training.recording.reports import TrainingResult
from src.training.recording.run_listener import (
    CsvRunListener,
    LoggingRunListener,
    RunListener,
)
from src.training.run_config import (
    ConfigDocument,
    ModelKind,
    ModelSpec,
    RunConfig,
    apply_overrides,
    parse_run_config,
    read_config_document,
)
from src.training.trainer import Trainer

_CONFIG_COPY_NAME = "run_config.yaml"
_ROUNDS_CSV_NAME = "rounds.csv"
_CHECKPOINTS_CSV_NAME = "checkpoints.csv"


def card_cost(card: GenericCard) -> int:
    """One card's share of HardwareLimits.max_batch_cost: 1, so the budget
    counts cards. Preflight and training both batch by it.

    Inputs: card. Output: 1. Side effects: none. Exceptions: none.

    Example:
        >>> card_cost(card)
        1
    """
    return 1


class ModelConstructor(Protocol):
    """A reference model class, built from its checkpoint and width."""

    def __call__(self, checkpoint: str, embed_dim: int) -> CardEncoderModel: ...


_MODEL_CONSTRUCTORS: Mapping[ModelKind, ModelConstructor] = {
    ModelKind.LINEAR_PROJECTION: LinearProjectionCardModel,
    ModelKind.RESIDUAL_MLP: ResidualMlpCardModel,
    ModelKind.ATTENTION_POOLING: AttentionPoolingCardModel,
}


@dataclass(frozen=True)
class CommandLine:
    """The parsed command line.

    config: the YAML run config. overrides: "dotted.key=value" strings, in
    order. check_only: build and preflight the dojos, build the model and
    validate the plan with a Trainer, then stop without training.
    """

    config: Path
    overrides: tuple[str, ...]
    check_only: bool


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
    args = parser.parse_args()
    return CommandLine(
        config=args.config, overrides=tuple(args.overrides), check_only=args.check
    )


def main() -> int:
    """Run one training run end to end; return a process exit code.

    Output: 0 on a completed run (even one that stopped early: the reason
        is printed) or a passed --check (preflight, model build and plan
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    command_line = parse_command_line()

    # Read, override and parse the config; fail fast before loading anything
    document = apply_overrides(
        read_config_document(command_line.config), command_line.overrides
    )
    config = parse_run_config(document)
    _require_device(config.device)
    if not command_line.check_only:
        _require_new_run_directory(config.run_directory)

    # Build the dojos and check each one before committing to a run
    context = DojoBuildContext(
        shelf=CardShelf(),
        holdout=config.plan.holdout,
        card_embedding_size=config.model.embed_dim,
        rng_seed=config.plan.seed,
        mod_overrides=config.mod_overrides,
        staple_thresholds=config.staple_thresholds,
        regression_objective=RegressionObjective.for_kind(config.regression_loss),
    )
    dojos = build_dojos(config.dojo_names, context)
    if not preflight_passes(dojos, config):
        return 1
    model = build_model(config.model).to(config.device)
    if command_line.check_only:
        # Constructing a Trainer validates the plan against the model and
        # dojos (e.g. a phase with nothing trainable) without training
        Trainer(model, dojos, config.plan, config.limits, None, [], cost_of=card_cost)
        print("Check passed; nothing trained.")
        return 0

    # Train, recording into a fresh run directory
    _create_run_directory(config.run_directory, document)
    result = train(model, dojos, config)
    print_summary(result, config.run_directory)
    return 0


def build_model(spec: ModelSpec) -> CardEncoderModel:
    """Construct the reference model spec names (on the CPU), through
    _MODEL_CONSTRUCTORS.

    Inputs: spec (ModelSpec). Output: CardEncoderModel.
    Side effects: may download the pretrained checkpoint.
    Exceptions: whatever the model constructor raises.

    Example:
        >>> build_model(ModelSpec(ModelKind.LINEAR_PROJECTION, "answerdotai/ModernBERT-base", 256))
    """
    constructor = _MODEL_CONSTRUCTORS[spec.kind]
    return constructor(checkpoint=spec.checkpoint, embed_dim=spec.embed_dim)


def preflight_passes(dojos: list[Dojo], config: RunConfig) -> bool:
    """Run preflight_dojo on every dojo and print one line each.

    Inputs: dojos (built dojos), config (for the batch budget and embed_dim).
    Output: True if every dojo passed.
    Side effects: prints a report; reads one TRAIN batch per dojo.
    Exceptions: none (a failing dojo is reported, not raised).

    Example:
        >>> preflight_passes(dojos, config)
        True
    """
    budget = BatchBudget(config.limits.max_batch_cost, card_cost)
    print(f"Preflight: {len(dojos)} dojos")
    passed = 0
    for dojo in dojos:
        check = preflight_dojo(dojo, budget, config.model.embed_dim)
        if check.ok:
            passed += 1
            print(
                f"[ OK ] {check.dojo_name}: train={check.train_count} "
                f"test={check.test_count} sample loss={check.sample_loss:.4g} "
                f"baseline={check.sample_baseline_loss:.4g}"
            )
        else:
            print(f"[FAIL] {check.dojo_name}: {check.error}")
        print_mod_tallies(check.mod_tallies)
    print(f"{passed}/{len(dojos)} dojos passed preflight")
    return passed == len(dojos)


def print_mod_tallies(tallies: Mapping[str, ModTally]) -> None:
    """One indented line per augmentation mod: cards seen, changed and
    failed over the preflight batch, with a WARNING on a mod that changed
    nothing (e.g. its keys no longer match the data) or failed on any card
    (with its first error). Informational: never fails preflight.

    Inputs: tallies (a PreflightResult's snapshot). Output: none.
    Side effects: prints. Exceptions: none.

    Example:
        >>> print_mod_tallies(check.mod_tallies)
               0:ShuffleKeysMod: changed 30/32, failed 0
    """
    for label, tally in tallies.items():
        line = (
            f"       {label}: changed {tally.cards_changed}/{tally.cards_seen}, "
            f"failed {tally.cards_failed}"
        )
        if tally.cards_failed:
            line += f"  WARNING: first failure {tally.first_failure}"
        elif tally.cards_seen and not tally.cards_changed:
            line += "  WARNING: changed no card (do its fields still match the data?)"
        print(line)


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


def print_summary(result: TrainingResult, run_directory: Path) -> None:
    """Print why the run ended, its final per-dojo TEST loss and where the
    best checkpoint and logs are.

    Inputs: result (TrainingResult), run_directory (Path). Output: none.
    Side effects: prints. Exceptions: none.

    Example:
        >>> print_summary(train(model, dojos, config), config.run_directory)
    """
    print(f"stopped early: {result.stopped_early_reason or 'no'}")
    if result.final_report is not None:
        print(f"final round {result.final_report.round_index} TEST loss:")
        for name, split_loss in sorted(result.final_report.per_dojo_test_loss.items()):
            print(
                f"  {name}: {split_loss.loss:.4g} "
                f"({split_loss.normalized:.3f}x baseline)"
            )
    if result.best_checkpoint is not None:
        print(f"best checkpoint: {result.best_checkpoint.path}")
    print(f"logs: {run_directory / _ROUNDS_CSV_NAME}")


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
