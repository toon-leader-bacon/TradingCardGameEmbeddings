"""One phase's mutable machinery, bundled so Trainer's methods stay short."""

from dataclasses import dataclass

import torch

from src.training.diet.diet_sampler import DietSampler
from src.training.diet.dojo_fault_ledger import DojoFaultLedger
from src.training.diet.dojo_batch_stream import DojoBatchStream
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
    """

    phase: Phase
    optimizer: torch.optim.Optimizer
    tracker: SaturationTracker
    sampler: DietSampler
    streams: dict[str, DojoBatchStream]
    faults: DojoFaultLedger
