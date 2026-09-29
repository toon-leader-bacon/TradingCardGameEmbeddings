"""Extrinsic evaluation: train fresh dojo heads on a frozen encoder and
record how well, and how fast, they learn.

An extrinsic run *is* a training run - one frozen phase of Trainer - so
training/ stays the one place that knows how to train; this module only
decides what to run and records the result. Compare encoders by calling
run_extrinsic once per encoder on the *same dojo objects*: reset_head()
restores every head to its as-built state, so each encoder starts from
identical heads.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import torch

from src.dojos.dojo import BatchBudget, Dojo
from src.encoder_model.precision import Precision
from src.schema.card import GenericCard
from src.schema.splits import Split
from src.training.plan import (
    DietRule,
    HardwareLimits,
    Phase,
    SaturationSpec,
    TrainingPlan,
)
from src.training.recording.reports import TrainingResult
from src.training.recording.run_listener import CsvRunListener
from src.training.round_evaluation import evaluate_split_losses
from src.training.trainable_encoder import TrainableEncoder
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

# Every row of an extrinsic rounds.csv carries this phase name
_PHASE_NAME = "extrinsic"
_ROUNDS_FILE = "rounds.csv"
_VALIDATION_FILE = "validation.json"


@dataclass(frozen=True)
class ExtrinsicSpec:
    """What to train, per encoder. Validated where it is used: invalid
    values raise ValueError from Phase / TrainingPlan construction inside
    run_extrinsic, before anything is written.

    diet_rule, head_lr, steps_per_round, max_rounds, saturation: the one
        frozen phase's settings (see training/plan.py Phase). A fixed
        budget: set saturation.patience_rounds above max_rounds.
    eval_examples_per_dojo: cap for both the per-round TEST pass and the
        final VALIDATION pass.
    seed: seeds Python random, torch, and the diet sampler.
    """

    diet_rule: DietRule
    head_lr: float
    steps_per_round: int
    max_rounds: int
    saturation: SaturationSpec
    eval_examples_per_dojo: int
    seed: int


@dataclass(frozen=True)
class ExtrinsicResult:
    """What one extrinsic run produced.

    rounds_csv: the per-round TEST losses (read with read_rounds_csv);
        None if the file does not exist after the run (no round completed,
        or the listener's writes failed - Trainer logs and skips them).
    validation_losses: final heads' VALIDATION loss per dojo (read-only);
        a dojo whose pass failed is omitted. None if the run stopped early
        (weights may be corrupt, or no head finished training) or the
        whole VALIDATION pass failed (logged) - then validation.json is not
        written either. If only writing validation.json failed (logged),
        the losses are still returned here.
    stopped_early_reason: from TrainingResult; None for a complete run.
    quarantined: dojos quarantined by the end of the run (empty if no
        round completed). Their VALIDATION loss, if any, is for an
        under-trained head.
    """

    rounds_csv: Path | None
    validation_losses: Mapping[str, float] | None
    stopped_early_reason: str | None
    quarantined: frozenset[str]

    def __post_init__(self) -> None:
        """Freeze validation_losses as a read-only copy (the dataclass is
        frozen). Inputs: none. Output: None. Side effects: none.
        Exceptions: none."""
        if self.validation_losses is not None:
            frozen = MappingProxyType(dict(self.validation_losses))
            object.__setattr__(self, "validation_losses", frozen)


def run_extrinsic(
    encoder: TrainableEncoder,
    encoder_label: str,
    dojos: Sequence[Dojo],
    spec: ExtrinsicSpec,
    hardware: HardwareLimits,
    output_dir: Path,
    *,
    cost_of: Callable[[GenericCard], int] = lambda card: 1,
) -> ExtrinsicResult:
    """Train every dojo's head on the frozen encoder, then score the final
    heads on VALIDATION.

    Two stages. (a) Validate, with no side effects: the output directory,
    this function's own checks, then Phase / TrainingPlan / CsvRunListener /
    Trainer construction (Trainer's constructor is the single source of
    plan-vs-dojo validation). A call that fails here leaves nothing on disk
    and can be retried. (b) Only then create the directory, seed, reset
    every head, run, and score.

    Inputs: encoder (any TrainableEncoder - a MultiCardModel on multi-card
        dojos is contextualized within each example); encoder_label (the
        run directory's name: one plain path segment); dojos (non-empty, every one with a trainable
        head, all sharing one holdout - the run's TrainingPlan.holdout;
        the encoder's own training holdout is never consulted); spec;
        hardware (batch budget and precision, for training and the
        VALIDATION pass alike); output_dir (the run writes into
        output_dir/extrinsic/<encoder_label>/); cost_of (per-card batch
        cost, passed to Trainer and used for the VALIDATION budget).
    Output: ExtrinsicResult.
    Side effects: creates output_dir/extrinsic/<encoder_label>/ (even if
        no round then completes: a partial run is a real result, so a
        retry needs a new label); writes rounds.csv and, unless stopped
        early or the VALIDATION pass failed, validation.json there;
        trains the dojos' heads; leaves the encoder frozen (requires_grad
        False) and in eval mode; reseeds the process-wide random and torch
        generators.
    Exceptions: FileExistsError if the run directory exists; ValueError if
        encoder_label is not one plain path segment, dojos is empty, a dojo
        has no trainable parameters, spec values are out of range (from
        Phase / TrainingPlan), or dojo names repeat or holdouts differ
        (from Trainer) - all before any side effect; OSError if the run
        directory cannot be created (still before training). Nothing
        raises after training starts: Trainer logs and skips failures,
        and a failed VALIDATION pass or validation.json write is logged
        and reflected in the result, so a finished run is never lost.

    Example:
        >>> for label, model in (("trained", trained), ("untrained", fresh)):
        ...     run_extrinsic(model, label, dojos, spec, hardware, out)
    """
    # (a) Validate everything; nothing touches disk or the dojos yet
    _require_run_label(encoder_label)
    run_dir = output_dir / "extrinsic" / encoder_label
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already holds an extrinsic run")
    _require_trainable_heads(dojos)
    plan = _build_plan(spec, dojos)
    listener = CsvRunListener(run_dir / _ROUNDS_FILE)
    trainer = Trainer(encoder, dojos, plan, hardware, None, [listener], cost_of=cost_of)

    # (b) Run from identical starting heads, reproducibly
    run_dir.mkdir(parents=True)
    _seed_everything(spec.seed)
    for dojo in dojos:
        dojo.reset_head()
    training = trainer.run()

    # Score the final heads unless the run stopped early; from here on a
    # failure is logged, never raised, so the finished run is kept
    validation: Mapping[str, float] | None = None
    quarantined = _collect_quarantined(training)
    if training.stopped_early_reason is None:
        budget = BatchBudget(hardware.max_batch_cost, cost_of)
        validation = _score_final_heads(
            encoder, dojos, budget, spec.eval_examples_per_dojo, hardware.precision
        )
    if validation is not None:
        _write_validation(run_dir / _VALIDATION_FILE, validation, quarantined)

    rounds_csv = run_dir / _ROUNDS_FILE
    return ExtrinsicResult(
        rounds_csv=rounds_csv if rounds_csv.exists() else None,
        validation_losses=validation,
        stopped_early_reason=training.stopped_early_reason,
        quarantined=quarantined,
    )


def _require_run_label(encoder_label: str) -> None:
    """Inputs: encoder_label. Output: None. Side effects: none.
    Exceptions: ValueError unless it is one plain path segment - non-empty,
    not "." or "..", no separator or null byte, not absolute - so the run directory
    stays inside output_dir/extrinsic/."""
    # Path(...).name differs from the label if it has a separator or is absolute
    is_segment = Path(encoder_label).name == encoder_label
    unusable = (
        not encoder_label or encoder_label in (".", "..") or "\0" in encoder_label
    )
    if unusable or not is_segment:
        raise ValueError(
            f"encoder_label must be one plain path segment, got {encoder_label!r}"
        )


def _score_final_heads(
    encoder: TrainableEncoder,
    dojos: Sequence[Dojo],
    budget: BatchBudget,
    max_examples: int,
    precision: Precision,
) -> Mapping[str, float] | None:
    """The VALIDATION pass over the trained heads, never raising.

    Inputs: the encoder, dojos, the batch budget, the per-dojo cap, the
        forward-pass precision.
    Output: evaluate_split_losses' dojo -> loss mapping (a failing dojo is
        already omitted there), or None if the pass as a whole failed (e.g.
        autocast unsupported on the device) - logged with its traceback.
    Side effects: forward passes (eval mode, no_grad); logs on failure.
    Exceptions: none (KeyboardInterrupt still propagates).
    """
    try:
        return evaluate_split_losses(
            encoder,
            dojos,
            split=Split.VALIDATION,
            budget=budget,
            max_examples=max_examples,
            precision=precision,
        )
    except Exception:
        logger.error("VALIDATION pass failed; the run is kept", exc_info=True)
        return None


def _require_trainable_heads(dojos: Sequence[Dojo]) -> None:
    """Inputs: dojos. Output: None. Side effects: none (trainable_parameters
    is iterated, not modified - GenericDojo returns a generator, always
    truthy, so it must be consumed). Exceptions: ValueError if dojos is
    empty or a dojo (named) yields no trainable parameters: a frozen
    encoder plus a head-less dojo (e.g. ContrastiveDojo) has nothing to
    train. Trainer's own check is per phase (it only refuses a phase
    whose every head is empty), so this per-dojo check is not redundant."""
    if not dojos:
        raise ValueError("an extrinsic run needs at least one dojo")
    for dojo in dojos:
        if next(iter(dojo.trainable_parameters()), None) is None:
            raise ValueError(
                f"dojo {dojo.name!r} has no trainable head to train on a frozen encoder"
            )


def _build_plan(spec: ExtrinsicSpec, dojos: Sequence[Dojo]) -> TrainingPlan:
    """The fixed spec -> plan mapping (not caller-configurable).

    Inputs: spec, dojos (non-empty).
    Output: TrainingPlan with one Phase(name="extrinsic", dojo_names in
        dojos' order, the spec's diet/lr/steps/rounds/saturation,
        encoder_trainable=False, encoder_lr=0.0, default max_grad_norm),
        holdout=dojos[0].holdout, no held-out dojos, the spec's
        eval_examples_per_dojo and seed, default FaultPolicy.
    Side effects: none.
    Exceptions: ValueError from Phase / TrainingPlan for out-of-range spec
        values.
    """
    phase = Phase(
        name=_PHASE_NAME,
        dojo_names=tuple(dojo.name for dojo in dojos),
        diet_rule=spec.diet_rule,
        encoder_trainable=False,
        encoder_lr=0.0,
        head_lr=spec.head_lr,
        steps_per_round=spec.steps_per_round,
        max_rounds=spec.max_rounds,
        saturation=spec.saturation,
    )
    return TrainingPlan(
        phases=(phase,),
        holdout=dojos[0].holdout,
        held_out_dojos=frozenset(),
        eval_examples_per_dojo=spec.eval_examples_per_dojo,
        seed=spec.seed,
    )


def _seed_everything(seed: int) -> None:
    """Inputs: seed. Output: None. Side effects: seeds Python's random and
    torch's global generators (mods draw on random; heads' dropout and any
    torch sampling on torch). Exceptions: none."""
    random.seed(seed)
    torch.manual_seed(seed)


def _collect_quarantined(training: TrainingResult) -> frozenset[str]:
    """Inputs: the TrainingResult. Output: the final report's quarantined
    dojos, or empty if no round completed. Side effects: none.
    Exceptions: none."""
    report = training.final_report
    return frozenset() if report is None else report.quarantined


def _write_validation(
    path: Path, losses: Mapping[str, float], quarantined: frozenset[str]
) -> None:
    """Inputs: the path, VALIDATION losses, the quarantined dojos. Output:
    None. Side effects: writes {"losses": {dojo: loss}, "quarantined":
    [sorted names]} as indented JSON (losses are finite: evaluate_split_losses
    omits a dojo with a non-finite loss); on an OSError, logs it and
    writes nothing more. Exceptions: none."""
    payload = {"losses": dict(losses), "quarantined": sorted(quarantined)}
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except OSError:
        logger.error("could not write %s; the losses are kept", path, exc_info=True)
