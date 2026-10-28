"""The training loop: phases -> rounds -> optimizer steps.

Knows only the Dojo Protocol, so generic and contrastive dojos are
interchangeable. One optimizer step trains on one batch from one dojo.
"""

import gc
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from src.dojos.dojo import BatchBudget, Dojo, DojoBatch
from src.encoder_model.precision import (
    autocast_for,
    device_type_of,
    parameter_device,
)
from src.schema.card import GenericCard
from src.schema.splits import Split
from src.training.diet.batch_feed import open_batch_feed
from src.training.diet.diet_sampler import diet_sampler_for
from src.training.diet.dojo_batch_stream import DojoBatchStream
from src.training.diet.dojo_fault_ledger import DojoFaultLedger
from src.training.diet.batch_feed_errors import BatchFeedError
from src.training.diet.prefetching_batch_streams import (
    NO_BATCH_WARMING,
    BatchWarmer,
)
from src.training.diet.round_schedule import RoundSchedule, draw_round_schedule
from src.training.diet.saturation_tracker import SaturationTracker
from src.training.gradient_clip_tally import GradientClipTally
from src.training.learning_rate_schedule import LearningRateSchedule
from src.training.phase_run import PhaseRun
from src.training.plan import HardwareLimits, Phase, TrainingPlan
from src.training.recording.checkpointer import Checkpointer
from src.training.recording.reports import (
    CheckpointRecord,
    RoundReport,
    TrainingResult,
    is_better_round,
    mean_normalized_test_loss_over,
)
from src.training.recording.run_listener import RunListener
from src.training.round_evaluation import evaluate_split_losses
from src.training.trainable_encoder import TrainableEncoder

logger = logging.getLogger(__name__)

# fp16 loss scale floor: overflow above it is the GradScaler probing its
# scale (skip the step, back off); overflow at it means the gradients are
# non-finite in true units, a real fault. Clamping also stops one bad dojo
# from driving the shared scale toward zero.
_MIN_LOSS_SCALE = 1.0

# Param-group key holding a group's rate before the phase's lr_schedule
_BASE_LR = "base_lr"


class NonFiniteLossError(ValueError):
    """A dojo produced a NaN or infinite loss; the step is skipped."""


class NonFiniteGradientError(ValueError):
    """fp16 gradients overflowed even at the minimum loss scale; the step is
    skipped."""


@dataclass
class _RoundSkips:
    """Steps that took no optimizer step this round, by cause. Neither is a
    fault: an fp16 overflow is the GradScaler probing its scale, and a batch
    with no usable baseline is a harmless degenerate batch."""

    fp16_overflow: int = 0
    unusable_baseline: int = 0

    def log_and_reset(self, phase_name: str, round_index: int) -> None:
        """Log (INFO) each nonzero count, then zero both.
        Inputs: phase_name, round_index (for the log line). Output: None.
        Side effects: logs; resets the counts. Exceptions: none."""
        if self.fp16_overflow:
            logger.info(
                "[%s] round %d: %d fp16 step(s) skipped on gradient overflow",
                phase_name,
                round_index,
                self.fp16_overflow,
            )
        if self.unusable_baseline:
            logger.info(
                "[%s] round %d: %d step(s) skipped, batch had no usable baseline",
                phase_name,
                round_index,
                self.unusable_baseline,
            )
        self.fp16_overflow = 0
        self.unusable_baseline = 0


@dataclass(frozen=True)
class _PhaseOutcome:
    """What one phase hands back to run().

    report: the phase's last RoundReport; None if it reported no round.
    best: its best checkpoint; None if it saved none.
    stopped_early_reason: why the whole run must stop now; None to go on
        to the next phase.
    """

    report: RoundReport | None
    best: CheckpointRecord | None
    stopped_early_reason: str | None


def _apply_learning_rate_schedule(phase_run: PhaseRun) -> None:
    """Set every optimizer group's lr to its base rate times the phase's
    lr_schedule multiplier at phase_run.step_attempts.

    Inputs: phase_run. Output: None.
    Side effects: writes each param group's "lr". Exceptions: none.
    """
    phase = phase_run.phase
    multiplier = phase.lr_schedule.multiplier(phase_run.step_attempts, phase.max_steps)
    for group in phase_run.optimizer.param_groups:
        group["lr"] = group[_BASE_LR] * multiplier


def _log_learning_rate(phase_run: PhaseRun, round_index: int) -> None:
    """Log (INFO) the schedule's multiplier at the end of a round, for a
    phase with a non-default lr_schedule only (a constant rate would repeat
    one line every round).

    Inputs: phase_run, round_index. Output: None.
    Side effects: logs. Exceptions: none.
    """
    phase = phase_run.phase
    if phase.lr_schedule == LearningRateSchedule():
        return
    multiplier = phase.lr_schedule.multiplier(phase_run.step_attempts, phase.max_steps)
    logger.info(
        "[%s] round %d: learning rate x%.3g of encoder_lr / head_lr next",
        phase.name,
        round_index,
        multiplier,
    )


class Trainer:
    """Trains one encoder against a suite of dojos according to a plan.

    Inputs (constructor): model, dojos (every registered dojo, including
        held-out ones), plan, limits, checkpointer (None: never save, e.g.
        evaluation's extrinsic runs, smoke tests, dry runs), listeners,
        cost_of (per-card batch cost; 1 per card by default, upgradable to
        token count without touching any dojo), batch_warmer (how the
        prefetch worker reserves each built batch's cards; nothing by
        default; used only when limits.prefetch_depth >= 1).
    Exceptions (constructor): ValueError if any dojo.holdout != plan.holdout,
        if a phase names an unregistered dojo, or if a phase's diet
        includes a held-out dojo.
    """

    def __init__(
        self,
        model: TrainableEncoder,
        dojos: Sequence[Dojo],
        plan: TrainingPlan,
        limits: HardwareLimits,
        checkpointer: Checkpointer | None,
        listeners: Sequence[RunListener],
        *,
        cost_of: Callable[[GenericCard], int] = lambda card: 1,
        batch_warmer: BatchWarmer = NO_BATCH_WARMING,
    ) -> None:
        self._model = model
        self._dojos = {dojo.name: dojo for dojo in dojos}
        self._plan = plan
        self._checkpointer = checkpointer
        self._listeners = listeners
        # One budget for every Dojo.batches() call (not yet written to the manifest)
        self._budget = BatchBudget(limits.max_batch_cost, cost_of)
        self._rng = random.Random(plan.seed)
        self._precision = limits.precision
        self._prefetch_depth = limits.prefetch_depth
        self._batch_warmer = batch_warmer
        self._steps_taken = 0
        # Every step attempt of the run, over every phase (read by the sampler)
        self._steps_attempted = 0
        # Steps skipped since the last round's report
        self._skips = _RoundSkips()
        # Per-dojo clipped steps since the last round's report
        self._clips = GradientClipTally()
        # The encoder parameters that are trainable as built (a pretrained
        # LM frozen by its own flag stays frozen); phases toggle only these
        self._encoder_params = [p for p in model.parameters() if p.requires_grad]
        self._validate_plan_against_dojos(dojos)

    @property
    def steps_attempted(self) -> int:
        """Step attempts so far, over every phase (any thread may read it).
        Inputs: none. Output: int. Side effects: none. Exceptions: none.

        Example:
            >>> trainer.steps_attempted
            1200
        """
        return self._steps_attempted

    def run(self) -> TrainingResult:
        """Execute every phase of the plan in order, surviving bad steps.

        Inputs: none (all configuration is in the constructor).
        Output: TrainingResult: the last round's report, the best
            checkpoint of the last phase that wrote one (best = strictly
            lowest mean normalized TEST loss over the diet dojos scored in
            both rounds),
            and why the run stopped early, if it did. best_checkpoint is
            always None when the Trainer has no checkpointer.
        Side effects: updates model and dojo-head weights; writes
            checkpoints (if a checkpointer was given); notifies listeners;
            logs every recovered failure.
        Exceptions: none for step, evaluation, checkpoint, listener or
            batch feed failures (logged and skipped; a feed error,
            BatchFeedError, ends the run early with that reason and is
            never counted against a dojo). Every phase's
            batch feed is closed when the phase ends, however it ends.
            KeyboardInterrupt propagates.

        Example:
            >>> result = Trainer(model, dojos, plan, limits, ckpt, []).run()
        """
        report: RoundReport | None = None
        last_good: CheckpointRecord | None = None
        stopped_early_reason: str | None = None
        run_started = time.monotonic()
        # Heads live in the dojos, not the model, so follow the model's device
        device = parameter_device(self._model)
        for dojo in self._dojos.values():
            dojo.move_head_to(device)
        # Train each phase; its feed's worker never outlives it
        for phase in self._plan.phases:
            phase_run = self._open_phase(phase)
            try:
                outcome = self._run_phase(phase_run, run_started)
            finally:
                phase_run.feed.close()
            # A phase that reported or saved nothing keeps the earlier ones
            report = outcome.report or report
            last_good = outcome.best or last_good
            if outcome.stopped_early_reason:
                stopped_early_reason = outcome.stopped_early_reason
                break
        return TrainingResult(report, last_good, stopped_early_reason)

    def _run_phase(self, phase_run: PhaseRun, run_started: float) -> _PhaseOutcome:
        """Train one phase's rounds: to saturation, its round cap, too many
        failures, every dojo quarantined, or a batch feed error. Each
        round's steps run on its own schedule, and the feed's round is
        ended before its evaluation and checkpoints.

        Inputs: phase_run (from _open_phase), run_started (time.monotonic()).
        Output: _PhaseOutcome.
        Side effects: as run(), for this phase; begins and ends rounds of
            phase_run.feed (does not close it).
        Exceptions: none for step, evaluation, checkpoint, listener or
            batch feed failures. KeyboardInterrupt propagates.

        Example:
            >>> outcome = self._run_phase(self._open_phase(phase), time.monotonic())
        """
        phase = phase_run.phase
        report: RoundReport | None = None
        best: CheckpointRecord | None = None
        for round_index in range(phase.max_rounds):
            # Steps: the worker builds ahead while the GPU trains
            try:
                self._train_round(phase_run)
            except BatchFeedError as error:
                return self._outcome_after_feed_error(phase_run, report, best, error)
            # Evaluation and checkpoints: the worker idle, dojos see only this thread
            phase_run.feed.end_round()
            self._skips.log_and_reset(phase.name, round_index)
            self._clips.log_and_reset(phase.name, round_index)
            _log_learning_rate(phase_run, round_index)
            if phase_run.faults.gave_up():
                # Keep the last good checkpoint: current weights may be corrupt
                reason = f"too many consecutive failures in {phase.name!r}"
                return _PhaseOutcome(report, best, reason)
            report = self._evaluate_round(phase_run, round_index, run_started)
            self._notify_round_end(report)
            best = self._checkpoint_if_best(best, phase_run, report)
            self._checkpoint_latest(phase_run, report)
            quarantined = phase_run.faults.quarantined_names()
            if quarantined.issuperset(phase.dojo_names):
                reason = f"every dojo quarantined in {phase.name!r}"
                return _PhaseOutcome(report, best, reason)
            if phase_run.tracker.phase_done(quarantined):
                break
        return _PhaseOutcome(report, best, None)

    def _outcome_after_feed_error(
        self,
        phase_run: PhaseRun,
        report: RoundReport | None,
        best: CheckpointRecord | None,
        error: BatchFeedError,
    ) -> _PhaseOutcome:
        """Log (ERROR, with any worker error chained) that the batch feed
        failed, and end the run keeping what this phase already has.
        Private helper - single consumer is _run_phase.

        Inputs: phase_run, report and best (this phase's so far), error.
        Output: _PhaseOutcome with stopped_early_reason naming the phase.
        Side effects: logs. Exceptions: none.
        """
        name = phase_run.phase.name
        logger.error("[%s] batch feed failed; ending the run", name, exc_info=error)
        return _PhaseOutcome(report, best, f"batch feed failed in {name!r}: {error}")

    def _validate_plan_against_dojos(self, dojos: Sequence[Dojo]) -> None:
        """Fail fast, before an overnight run starts, if the plan and dojos
        disagree.

        Raises ValueError if dojo names repeat, any dojo.holdout !=
        plan.holdout, a phase or held-out name is unregistered, or a phase
        has nothing to train (frozen encoder and no dojo head parameters).
        """
        if len(self._dojos) != len(dojos):
            raise ValueError("dojo names must be unique")
        for dojo in dojos:
            if dojo.holdout != self._plan.holdout:
                raise ValueError(f"dojo {dojo.name!r} holdout differs from the plan's")
        named = set(self._plan.held_out_dojos)
        for phase in self._plan.phases:
            named.update(phase.dojo_names)
        unknown = named - set(self._dojos)
        if unknown:
            raise ValueError(f"plan names unregistered dojos: {sorted(unknown)}")
        for phase in self._plan.phases:
            has_encoder = phase.encoder_trainable and bool(self._encoder_params)
            if not has_encoder and not self._head_parameters(phase):
                raise ValueError(f"phase {phase.name!r} has no trainable parameters")
            self._build_optimizer(phase)  # surface optimizer errors now

    def _open_phase(self, phase: Phase) -> PhaseRun:
        """Build the optimizer (two param groups; encoder frozen if not
        encoder_trainable), tracker, sampler, batch feed (direct, or
        prefetching at limits.prefetch_depth >= 1; the caller must close
        it) and fault ledger (from plan.faults)."""
        for parameter in self._encoder_params:
            parameter.requires_grad_(phase.encoder_trainable)
        optimizer = self._build_optimizer(phase)
        self._log_trainable_parameters(phase)
        streams = {
            name: DojoBatchStream(self._dojos[name], self._budget)
            for name in phase.dojo_names
        }
        return PhaseRun(
            phase=phase,
            optimizer=optimizer,
            tracker=SaturationTracker(phase.saturation, phase.dojo_names),
            sampler=diet_sampler_for(phase.diet_rule),
            feed=open_batch_feed(streams, self._batch_warmer, self._prefetch_depth),
            faults=DojoFaultLedger(self._plan.faults, phase.dojo_names),
            scaler=torch.amp.GradScaler(
                device_type_of(self._model), enabled=self._precision == "fp16"
            ),
            weighting=self._plan.loss_weighting,
        )

    def _log_trainable_parameters(self, phase: Phase) -> None:
        """Log how many parameters phase trains, per optimizer group, so a
        config that trains nothing useful is obvious from the first line.
        Inputs: phase. Output: None. Side effects: logs at INFO.
        Exceptions: none."""
        encoder_count = 0
        if phase.encoder_trainable:
            encoder_count = sum(p.numel() for p in self._encoder_params)
        head_count = sum(p.numel() for p in self._head_parameters(phase))
        logger.info(
            "[%s] trainable parameters: encoder %d (lr %g), dojo heads %d (lr %g); "
            "lr schedule %s",
            phase.name,
            encoder_count,
            phase.encoder_lr,
            head_count,
            phase.head_lr,
            phase.lr_schedule,
        )

    def _build_optimizer(self, phase: Phase) -> torch.optim.Optimizer:
        """AdamW with an encoder group at encoder_lr (only if trainable)
        and a head group at head_lr; empty groups are omitted."""
        # _BASE_LR holds the rate before lr_schedule's multiplier (torch keeps
        # extra keys in a param group)
        groups: list[dict[str, Any]] = []
        if phase.encoder_trainable and self._encoder_params:
            groups.append(
                {
                    "params": self._encoder_params,
                    "lr": phase.encoder_lr,
                    _BASE_LR: phase.encoder_lr,
                }
            )
        head_parameters = self._head_parameters(phase)
        if head_parameters:
            groups.append(
                {
                    "params": head_parameters,
                    "lr": phase.head_lr,
                    _BASE_LR: phase.head_lr,
                }
            )
        return torch.optim.AdamW(groups)

    def _head_parameters(self, phase: Phase) -> list[torch.nn.Parameter]:
        """Every phase dojo's trainable_parameters, each parameter once."""
        seen: dict[int, torch.nn.Parameter] = {}
        encoder_ids = {id(parameter) for parameter in self._encoder_params}
        for name in phase.dojo_names:
            for parameter in self._dojos[name].trainable_parameters():
                if id(parameter) not in encoder_ids:
                    seen.setdefault(id(parameter), parameter)
        return list(seen.values())

    def _active_dojos(self, phase_run: PhaseRun) -> list[Dojo]:
        """The tracker's ACTIVE dojos minus any the ledger quarantined."""
        return [
            self._dojos[name]
            for name in phase_run.tracker.active_dojo_names()
            if not phase_run.faults.is_quarantined(name)
        ]

    def _train_round(self, phase_run: PhaseRun) -> None:
        """Take steps_per_round optimizer steps on the phase's ACTIVE dojos,
        in the order of a schedule drawn when the round starts.

        A failing step (exception, out-of-memory, non-finite loss) is
        logged, skipped and counted; it never ends the run by itself. A
        quarantine mid-round redraws the rest of the round from the dojos
        still active and reschedules the feed.

        Inputs: phase_run (the phase's optimizer, tracker, sampler, feed,
            fault ledger).
        Output: None; returns early if every dojo is quarantined, the
            ledger gave_up(), or no schedule can be drawn.
        Side effects: updates model and the sampled dojos' head weights;
            advances self._rng and the sampler (all of a round's draws at
            its start, the rest of it again after a quarantine); begins a
            round of the batch feed and takes its batches; logs.
        Exceptions: none for step failures. BatchFeedError propagates (a
            lifecycle error, never counted against a dojo).
            KeyboardInterrupt propagates.

        Example:
            >>> self._train_round(phase_run)
        """
        faults = phase_run.faults
        steps = phase_run.phase.steps_per_round
        # A frozen encoder stays in eval mode (no dropout) while heads train;
        # dojo heads are not part of the model, so this does not touch them
        self._model.train(phase_run.phase.encoder_trainable)
        active = self._active_dojos(phase_run)
        schedule = self._draw_schedule(phase_run, active, steps)
        if schedule is None:
            return
        phase_run.feed.begin_round(schedule)
        names = list(schedule.dojo_names)

        # One optimizer step per scheduled step, each on one batch
        for step_index in range(steps):
            self._attempt_step(phase_run, self._dojos[names[step_index]])
            if faults.gave_up():
                return

            # A quarantine changes the diet: redraw the rest of the round
            still_active = self._active_dojos(phase_run)
            if still_active != active:
                active = still_active
                rest = self._draw_schedule(phase_run, active, steps - step_index - 1)
                if rest is None:
                    return
                phase_run.feed.reschedule(rest)
                names[step_index + 1 :] = rest.dojo_names

    def _attempt_step(self, phase_run: PhaseRun, dojo: Dojo) -> None:
        """One scheduled step: advance the learning-rate schedule, take the
        step, and record its success or recover from its failure.

        Private helper - single consumer is _train_round.
        Inputs: phase_run, dojo (the scheduled one). Output: none.
        Side effects: as _take_step; step_attempts grows; the fault ledger
            records the outcome (a skipped degenerate batch is neither).
        Exceptions: BatchFeedError propagates; any other failure is
            recovered from.
        """
        failure: Exception | None = None
        ran = False
        # Every attempt advances the schedule, so it follows the budget
        _apply_learning_rate_schedule(phase_run)
        phase_run.step_attempts += 1
        self._steps_attempted += 1
        try:
            ran = self._take_step(phase_run, dojo)
        except BatchFeedError:
            raise  # not a step failure: _run_phase ends the run
        except Exception as error:
            failure = error
        # Recover outside the except block: inside it the traceback still
        # pins the failed step's tensors, so an out-of-memory would not free
        if failure is not None:
            self._recover_from_failed_step(phase_run, dojo, failure)
        elif ran:
            phase_run.faults.record_success(dojo.name)

    def _draw_schedule(
        self, phase_run: PhaseRun, active: list[Dojo], steps: int
    ) -> RoundSchedule | None:
        """The next steps of the round, drawn from the phase's sampler; None
        when there is nothing to draw (no active dojo, no step left) or
        the sampler fails. A sampler failure is recorded as one failure
        with no dojo, uses no step attempt, and ends the round: a sampler
        that keeps failing reaches gave_up() after that many rounds.

        Private helper - single consumer is _train_round.
        Inputs: phase_run, active, steps (>= 0). Output: RoundSchedule or None.
        Side effects: advances self._rng and the sampler; on a sampler
            failure, records it in the fault ledger.
        Exceptions: none.
        """
        if not active or steps == 0:
            return None
        failure: Exception
        try:
            return draw_round_schedule(phase_run.sampler, active, steps, self._rng)
        except Exception as error:
            failure = error
        # Recover outside the except block, as for a failed step
        self._recover_from_failed_step(phase_run, None, failure)
        return None

    def _take_step(self, phase_run: PhaseRun, dojo: Dojo) -> bool:
        """One optimizer step on the dojo's next batch. Raises on any
        failure (caught only by _train_round).

        Output: True when the step ran to completion (an fp16 overflow skip
        counts: it is normal). False when the batch had no usable baseline
        and the step was skipped before any forward pass: that is counted
        in self._skips and is neither a success nor a fault.
        """
        batch = phase_run.feed.next_batch(dojo.name)

        # Scale first: a degenerate batch costs no forward pass
        scale = self._loss_scale_for(phase_run, dojo, batch)
        if scale is None:
            self._skips.unusable_baseline += 1
            return False

        # Forward: encoder embeds batch.inputs, the dojo scores them
        with autocast_for(self._model, self._precision):
            loss = self._loss_of(dojo, batch)
        self._require_finite(loss, dojo)
        loss = loss * scale

        # Backward (loss scaled under fp16, else unchanged), then unscale so
        # clipping sees gradients in true units
        optimizer, scaler = phase_run.optimizer, phase_run.scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if not scaler.is_enabled():
            # A non-finite gradient raises here, skipping the step instead
            # of poisoning the weights and Adam state
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, phase_run.phase.max_grad_norm, error_if_nonfinite=True
            )
            self._clips.record(dojo.name, norm.item(), phase_run.phase.max_grad_norm)
            optimizer.step()
            self._steps_taken += 1
            return True
        norm = torch.nn.utils.clip_grad_norm_(parameters, phase_run.phase.max_grad_norm)
        # An fp16 overflow's norm is non-finite, and the tally skips it
        self._clips.record(dojo.name, norm.item(), phase_run.phase.max_grad_norm)
        self._scaled_step(scaler, optimizer, dojo)
        return True

    def _scaled_step(
        self, scaler: torch.amp.GradScaler, optimizer: torch.optim.Optimizer, dojo: Dojo
    ) -> None:
        """Step through the scaler, which skips the step and backs off its
        scale if any unscaled gradient is non-finite. Such a skip is normal
        above _MIN_LOSS_SCALE; at it, the scale is held there and
        NonFiniteGradientError raises so the fault ledger counts it."""
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # The scale only shrinks when the step was skipped
        if scaler.get_scale() >= scale_before:
            self._steps_taken += 1
            return
        if scale_before > _MIN_LOSS_SCALE:
            self._skips.fp16_overflow += 1
            logger.debug(
                "fp16 overflow on %r; skipped the step, loss scale %g -> %g",
                dojo.name,
                scale_before,
                scaler.get_scale(),
            )
            return
        scaler.update(new_scale=_MIN_LOSS_SCALE)
        raise NonFiniteGradientError(
            f"dojo {dojo.name!r} gradients are non-finite at loss scale "
            f"{_MIN_LOSS_SCALE:g}"
        )

    def _loss_scale_for(
        self, phase_run: PhaseRun, dojo: Dojo, batch: DojoBatch
    ) -> float | None:
        """The factor _take_step multiplies this batch's loss by.

        Inputs: phase_run (carries the plan's LossWeighting or None), dojo,
            batch (one dojo yielded).
        Output: float | None. 1.0 when phase_run.weighting is None (train
            on the raw loss). Otherwise weighting.scale_for(dojo, batch),
            whose None means "this batch has no usable baseline": the
            caller skips the step, which is neither a success nor a fault.
        Side effects: none. Exceptions: TypeError from a mismatched batch.
        """
        if phase_run.weighting is None:
            return 1.0
        return phase_run.weighting.scale_for(dojo, batch)

    def _loss_of(self, dojo: Dojo, batch: DojoBatch) -> torch.Tensor:
        """Encode batch.inputs and return dojo.compute_loss (a scalar mean)."""
        embeddings = self._model(batch.inputs)
        return dojo.compute_loss(embeddings, batch)

    def _require_finite(self, loss: torch.Tensor, dojo: Dojo) -> None:
        """Raise NonFiniteLossError naming the dojo if loss is NaN or inf."""
        if not torch.isfinite(loss).all():
            raise NonFiniteLossError(f"dojo {dojo.name!r} loss is {loss.tolist()}")

    def _recover_from_failed_step(
        self, phase_run: PhaseRun, dojo: Dojo | None, error: Exception
    ) -> None:
        """Count the failure (dojo is None if selection itself failed), drop
        half-built gradients and free memory so an out-of-memory step does
        not cascade. Never raises: a CUDA error after an OOM is logged."""
        phase_run.faults.record_failure(dojo.name if dojo else None, error)
        error.__traceback__ = None  # release the failed step's tensors
        try:
            phase_run.optimizer.zero_grad(set_to_none=True)
            # Forget a half-finished unscale_ so the next step's unscale_
            # does not raise; the scale itself is kept
            phase_run.scaler.update(new_scale=phase_run.scaler.get_scale())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.error("cleanup after a failed step also failed", exc_info=True)

    def _evaluate_round(
        self, phase_run: PhaseRun, round_index: int, run_started: float
    ) -> RoundReport:
        """The TEST pass over ALL dojos, feed the tracker, assemble the
        RoundReport (elapsed_seconds measured from run_started, a
        time.monotonic() reading)."""
        losses = evaluate_split_losses(
            self._model,
            list(self._dojos.values()),
            split=Split.TEST,
            budget=self._budget,
            max_examples=self._plan.eval_examples_per_dojo,
            precision=self._precision,
        )
        # The tracker judges normalized loss, so epsilon/reactivation_delta
        # mean the same thing for every dojo
        statuses = phase_run.tracker.record_round(
            {name: split_loss.normalized for name, split_loss in losses.items()}
        )
        return RoundReport(
            phase=phase_run.phase.name,
            round_index=round_index,
            step=self._steps_taken,
            elapsed_seconds=time.monotonic() - run_started,
            per_dojo_test_loss=losses,
            statuses=statuses,
            quarantined=phase_run.faults.quarantined_names(),
            held_out=self._plan.held_out_dojos,
        )

    def _notify_round_end(self, report: RoundReport) -> None:
        """Call every listener's on_round_end; a raising listener is logged
        and skipped so it cannot stop the run."""
        for listener in self._listeners:
            try:
                listener.on_round_end(report)
            except Exception:
                logger.warning("listener failed in on_round_end", exc_info=True)

    def _checkpoint_latest(self, phase_run: PhaseRun, report: RoundReport) -> None:
        """Save this round as the run's latest checkpoint. Without a
        checkpointer, nothing; a failing save is logged, never raised.
        Listeners are not notified (the checkpoints CSV records bests)."""
        if self._checkpointer is None:
            return
        try:
            self._checkpointer.save_latest(
                self._model,
                list(self._dojos.values()),
                phase_run.optimizer,
                self._plan,
                report,
            )
        except Exception:
            logger.error("latest checkpoint save failed", exc_info=True)

    def _checkpoint_if_best(
        self,
        best: CheckpointRecord | None,
        phase_run: PhaseRun,
        report: RoundReport,
    ) -> CheckpointRecord | None:
        """Save and notify listeners when report beats best on the diet dojos
        both scored (or is the first round with any diet dojo scored);
        otherwise return best. A failing save or listener is
        logged and best is returned unchanged. Without a checkpointer,
        nothing is saved and None is returned."""
        if self._checkpointer is None:
            return None
        names = phase_run.phase.dojo_names
        if best is None:
            # Never checkpoint a round in which no diet dojo was scored
            if mean_normalized_test_loss_over(report, names) == float("inf"):
                return None
        elif not is_better_round(report, best.report, names):
            return best
        try:
            record = self._checkpointer.save(
                self._model,
                list(self._dojos.values()),
                phase_run.optimizer,
                self._plan,
                report,
            )
        except Exception:
            logger.error("checkpoint save failed", exc_info=True)
            return best
        for listener in self._listeners:
            try:
                listener.on_checkpoint(record)
            except Exception:
                logger.warning("listener failed in on_checkpoint", exc_info=True)
        return record
