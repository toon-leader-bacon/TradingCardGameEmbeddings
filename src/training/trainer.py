"""The training loop: phases -> rounds -> optimizer steps.

Knows only the Dojo Protocol, so generic and contrastive dojos are
interchangeable. One optimizer step trains on one batch from one dojo.
"""

import contextlib
import gc
import logging
import random
from typing import Any, Callable, ContextManager, Sequence

import torch

from src.dojos.dojo import BatchBudget, Dojo, DojoBatch
from src.schema.card import GenericCard
from src.training.diet.diet_sampler import diet_sampler_for
from src.training.diet.dojo_batch_stream import DojoBatchStream
from src.training.diet.dojo_fault_ledger import DojoFaultLedger
from src.training.diet.saturation_tracker import SaturationTracker
from src.training.phase_run import PhaseRun
from src.training.recording.checkpointer import Checkpointer
from src.training.plan import HardwareLimits, Phase, TrainingPlan
from src.training.recording.reports import (
    CheckpointRecord,
    RoundReport,
    TrainingResult,
    is_better_round,
    mean_test_loss_over,
)
from src.training.round_evaluation import evaluate_test_losses
from src.training.recording.run_listener import RunListener
from src.training.trainable_encoder import TrainableEncoder

logger = logging.getLogger(__name__)

# fp16 loss scale floor: overflow above it is the GradScaler probing its
# scale (skip the step, back off); overflow at it means the gradients are
# non-finite in true units, a real fault. Clamping also stops one bad dojo
# from driving the shared scale toward zero.
_MIN_LOSS_SCALE = 1.0

_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


class NonFiniteLossError(ValueError):
    """A dojo produced a NaN or infinite loss; the step is skipped."""


class NonFiniteGradientError(ValueError):
    """fp16 gradients overflowed even at the minimum loss scale; the step is
    skipped."""


class Trainer:
    """Trains one encoder against a suite of dojos according to a plan.

    Inputs (constructor): model, dojos (every registered dojo, including
        held-out ones), plan, limits, checkpointer, listeners, cost_of
        (per-card batch cost; 1 per card by default, upgradable to token
        count without touching any dojo).
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
        checkpointer: Checkpointer,
        listeners: Sequence[RunListener],
        *,
        cost_of: Callable[[GenericCard], int] = lambda card: 1,
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
        self._steps_taken = 0
        # The encoder parameters that are trainable as built (a pretrained
        # LM frozen by its own flag stays frozen); phases toggle only these
        self._encoder_params = [p for p in model.parameters() if p.requires_grad]
        self._validate_plan_against_dojos(dojos)

    def run(self) -> TrainingResult:
        """Execute every phase of the plan in order, surviving bad steps.

        Inputs: none (all configuration is in the constructor).
        Output: TrainingResult: the last round's report, the best
            checkpoint of the last phase that wrote one (best = strictly
            lowest mean TEST loss over the diet dojos scored in both rounds),
            and why the run stopped early, if it did.
        Side effects: updates model and dojo-head weights; writes
            checkpoints; notifies listeners; logs every recovered failure.
        Exceptions: none for step, evaluation, checkpoint or listener
            failures (logged and skipped). KeyboardInterrupt propagates.

        Example:
            >>> result = Trainer(model, dojos, plan, limits, ckpt, []).run()
        """
        report: RoundReport | None = None
        last_good: CheckpointRecord | None = None
        stopped_early_reason: str | None = None
        # Train each phase to saturation, its round cap, or too many failures
        for phase in self._plan.phases:
            phase_run = self._open_phase(phase)
            best: CheckpointRecord | None = None  # best is per phase
            for round_index in range(phase.max_rounds):
                self._train_round(phase_run)
                if phase_run.faults.gave_up():
                    # Keep the last good checkpoint: current weights may be corrupt
                    stopped_early_reason = (
                        f"too many consecutive failures in {phase.name!r}"
                    )
                    break
                report = self._evaluate_round(phase_run, round_index)
                self._notify_round_end(report)
                best = self._checkpoint_if_best(best, phase_run, report)
                quarantined = phase_run.faults.quarantined_names()
                if quarantined.issuperset(phase.dojo_names):
                    stopped_early_reason = f"every dojo quarantined in {phase.name!r}"
                    break
                if phase_run.tracker.phase_done(quarantined):
                    break
            # A phase that gave up before any checkpoint keeps the earlier one
            last_good = best or last_good
            if stopped_early_reason:
                break
        return TrainingResult(report, last_good, stopped_early_reason)

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
        encoder_trainable), tracker, sampler, batch streams and fault
        ledger (from plan.faults)."""
        for parameter in self._encoder_params:
            parameter.requires_grad_(phase.encoder_trainable)
        return PhaseRun(
            phase=phase,
            optimizer=self._build_optimizer(phase),
            tracker=SaturationTracker(phase.saturation, phase.dojo_names),
            sampler=diet_sampler_for(phase.diet_rule),
            streams={
                name: DojoBatchStream(self._dojos[name], self._budget)
                for name in phase.dojo_names
            },
            faults=DojoFaultLedger(self._plan.faults, phase.dojo_names),
            scaler=torch.amp.GradScaler(
                self._device_type(), enabled=self._precision == "fp16"
            ),
        )

    def _device_type(self) -> str:
        """The model's device type ("cuda" also on ROCm), read each call so
        a model moved after construction is followed."""
        for parameter in self._model.parameters():
            return parameter.device.type
        return "cpu"

    def _autocast(self) -> ContextManager[Any]:
        """torch.autocast at the configured precision; a no-op for fp32."""
        dtype = _AUTOCAST_DTYPES.get(self._precision)
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self._device_type(), dtype=dtype)

    def _build_optimizer(self, phase: Phase) -> torch.optim.Optimizer:
        """AdamW with an encoder group at encoder_lr (only if trainable)
        and a head group at head_lr; empty groups are omitted."""
        groups: list[dict[str, Any]] = []
        if phase.encoder_trainable and self._encoder_params:
            groups.append({"params": self._encoder_params, "lr": phase.encoder_lr})
        head_parameters = self._head_parameters(phase)
        if head_parameters:
            groups.append({"params": head_parameters, "lr": phase.head_lr})
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
        """Take steps_per_round optimizer steps on the phase's ACTIVE dojos.

        A failing step (exception, out-of-memory, non-finite loss) is
        logged, skipped and counted; it never ends the run by itself.

        Inputs: phase_run (the phase's optimizer, tracker, sampler,
            streams, fault ledger).
        Output: None; returns early if every dojo is quarantined or the
            ledger gave_up().
        Side effects: updates model and the sampled dojos' head weights;
            advances self._rng, the sampler and the batch streams; logs.
        Exceptions: none for step failures. KeyboardInterrupt propagates.

        Example:
            >>> self._train_round(phase_run)
        """
        faults = phase_run.faults
        self._model.train(True)
        active = self._active_dojos(phase_run)

        # One optimizer step per iteration, each on one batch from one dojo
        for _ in range(phase_run.phase.steps_per_round):
            if not active:
                return
            dojo: Dojo | None = None
            failure: Exception | None = None
            try:
                dojo = phase_run.sampler.next_dojo(active, self._rng)
                self._take_step(phase_run, dojo)
            except Exception as error:
                failure = error
            # Recover outside the except block: inside it the traceback still
            # pins the failed step's tensors, so an out-of-memory would not free
            if failure is not None:
                self._recover_from_failed_step(phase_run, dojo, failure)
                failure = None
            elif dojo is not None:
                faults.record_success(dojo.name)
            if faults.gave_up():
                return
            # A quarantine may have changed the diet
            active = self._active_dojos(phase_run)

    def _take_step(self, phase_run: PhaseRun, dojo: Dojo) -> None:
        """One optimizer step on the dojo's next batch. Raises on any
        failure (caught only by _train_round)."""
        batch = phase_run.streams[dojo.name].next_batch()

        # Forward: encoder embeds batch.inputs, the dojo scores them
        with self._autocast():
            loss = self._loss_of(dojo, batch)
        self._require_finite(loss, dojo)

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
            torch.nn.utils.clip_grad_norm_(
                parameters, phase_run.phase.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            self._steps_taken += 1
            return
        torch.nn.utils.clip_grad_norm_(parameters, phase_run.phase.max_grad_norm)
        self._scaled_step(scaler, optimizer, dojo)

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

    def _evaluate_round(self, phase_run: PhaseRun, round_index: int) -> RoundReport:
        """evaluate_test_losses over ALL dojos, feed the tracker, assemble
        the RoundReport."""
        with self._autocast():
            losses = evaluate_test_losses(
                self._model,
                list(self._dojos.values()),
                self._budget,
                self._plan.eval_examples_per_dojo,
            )
        statuses = phase_run.tracker.record_round(losses)
        return RoundReport(
            phase=phase_run.phase.name,
            round_index=round_index,
            step=self._steps_taken,
            per_dojo_test_loss=losses,
            statuses=statuses,
            quarantined=phase_run.faults.quarantined_names(),
        )

    def _notify_round_end(self, report: RoundReport) -> None:
        """Call every listener's on_round_end; a raising listener is logged
        and skipped so it cannot stop the run."""
        for listener in self._listeners:
            try:
                listener.on_round_end(report)
            except Exception:
                logger.warning("listener failed in on_round_end", exc_info=True)

    def _checkpoint_if_best(
        self,
        best: CheckpointRecord | None,
        phase_run: PhaseRun,
        report: RoundReport,
    ) -> CheckpointRecord | None:
        """Save and notify listeners when report beats best on the diet dojos
        both scored (or is the first round with any diet dojo scored);
        otherwise return best. A failing save or listener is
        logged and best is returned unchanged."""
        names = phase_run.phase.dojo_names
        if best is None:
            # Never checkpoint a round in which no diet dojo was scored
            if mean_test_loss_over(report, names) == float("inf"):
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
