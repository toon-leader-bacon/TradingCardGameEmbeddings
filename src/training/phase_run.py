"""One phase's mutable machinery, bundled so Trainer's methods stay short."""

from dataclasses import dataclass

import torch

from src.training.diet.diet_sampler import DietSampler
from src.training.diet.dojo_fault_ledger import DojoFaultLedger
from src.training.diet.dojo_batch_stream import DojoBatchStream
from src.training.loss_weighting import LossWeighting
from src.training.plan import Phase
from src.training.diet.saturation_tracker import SaturationTracker


@dataclass
class PhaseRun:
    """Everything Trainer needs to advance one phase.

    optimizer: encoder (if trainable) at encoder_lr + phase dojos' heads at head_lr.
    tracker: saturation state of the phase's dojos.
    sampler: the phase's diet policy.
    streams: dojo name -> endless TRAIN batches.
    faults: failure counts and dojo quarantine.
    scaler: fp16 loss scaler (a disabled pass-through at other precisions);
        one per phase, like the optimizer it steps.
    weighting: the plan's loss weighting, applied to every step's loss;
        None trains on the raw loss.
    step_attempts: optimizer steps attempted so far this phase, including
        skipped and failed ones: the step lr_schedule is evaluated at, so
        the schedule follows the step budget whatever fails.
    """

    phase: Phase
    optimizer: torch.optim.Optimizer
    tracker: SaturationTracker
    sampler: DietSampler
    streams: dict[str, DojoBatchStream]
    faults: DojoFaultLedger
    scaler: torch.amp.GradScaler
    weighting: LossWeighting | None
    step_attempts: int = 0
