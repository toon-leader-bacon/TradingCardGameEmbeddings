"""Experiment definition: what to train, in which phases, on which cards.

Everything here is a frozen, hardware-independent value so a run can be
serialized into its manifest and replayed. Hardware limits live apart in
HardwareLimits because they change per machine, not per experiment (they
affect results but are not yet written to the checkpoint manifest).
"""

import re
from dataclasses import dataclass

from src.schema.holdout import HoldoutSpec


@dataclass(frozen=True)
class Proportional:
    """Sample dojos in proportion to their TRAIN example count."""


@dataclass(frozen=True)
class Uniform:
    """Sample every active dojo equally often."""


@dataclass(frozen=True)
class Temperature:
    """Sample proportional to example_count ** alpha (alpha=1 is
    Proportional, alpha=0 is Uniform)."""

    alpha: float

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")


DietRule = Proportional | Uniform | Temperature


@dataclass(frozen=True)
class SaturationSpec:
    """When a dojo stops being worth training on.

    epsilon: a round counts as an improvement only if TEST loss drops by
        more than this below the dojo's best so far.
    patience_rounds: consecutive non-improving rounds before SATURATED.
    reactivation_delta: a SATURATED dojo re-enters the diet if its TEST
        loss rises this far above its best.
    target_saturated_fraction: the phase exits when this fraction of its
        dojos is SATURATED (in [0, 1]).
    """

    epsilon: float
    patience_rounds: int
    reactivation_delta: float
    target_saturated_fraction: float

    def __post_init__(self) -> None:
        if self.patience_rounds < 1:
            raise ValueError("patience_rounds must be >= 1")
        if not 0 <= self.target_saturated_fraction <= 1:
            raise ValueError("target_saturated_fraction must be in [0, 1]")


@dataclass(frozen=True)
class FaultPolicy:
    """How an unattended run reacts to failures instead of crashing.

    A failed step (exception, out-of-memory, non-finite loss) is logged and
    skipped. max_consecutive_dojo_failures: failures in a row before that
    dojo is quarantined for the rest of the phase.
    max_total_dojo_failures: failures in a phase, consecutive or not,
    before the dojo is quarantined (catches a dojo that fails every pass at
    the same row but succeeds in between).
    max_consecutive_failures: failures in a row across all dojos before the
    run stops cleanly and returns its best checkpoint.
    """

    max_consecutive_dojo_failures: int = 5
    max_consecutive_failures: int = 20
    max_total_dojo_failures: int = 50

    def __post_init__(self) -> None:
        limits = (
            self.max_consecutive_dojo_failures,
            self.max_consecutive_failures,
            self.max_total_dojo_failures,
        )
        if any(limit < 1 for limit in limits):
            raise ValueError("fault limits must be >= 1")


_PHASE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Phase:
    """One stage of training (pre-train, post-train, curriculum step...).

    dojo_names: the dojos in this phase's diet; every name must match a
        registered Dojo.name and none may be in TrainingPlan.held_out_dojos.
    encoder_trainable: False freezes the whole encoder (head-only stage).
    encoder_lr / head_lr: learning rates of the encoder and of every
        dojo's decoder head.
    steps_per_round: optimizer steps between diet re-decisions/evaluations.
    max_rounds: hard cap on rounds if saturation is never reached.
    max_grad_norm: gradients are clipped to this norm; a non-finite
        gradient skips the step instead of poisoning the weights.
    """

    name: str
    dojo_names: tuple[str, ...]
    diet_rule: DietRule
    encoder_trainable: bool
    encoder_lr: float
    head_lr: float
    steps_per_round: int
    max_rounds: int
    saturation: SaturationSpec
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.steps_per_round < 1 or self.max_rounds < 1:
            raise ValueError("steps_per_round and max_rounds must be >= 1")
        if not self.dojo_names:
            raise ValueError("a phase needs at least one dojo")
        if len(set(self.dojo_names)) != len(self.dojo_names):
            raise ValueError("a phase lists a dojo more than once")
        # `not x >= 0` also rejects NaN
        if not (self.encoder_lr >= 0 and self.head_lr >= 0 and self.max_grad_norm > 0):
            raise ValueError("learning rates must be >= 0 and max_grad_norm > 0")
        if not _PHASE_NAME.fullmatch(self.name):
            raise ValueError(f"phase name {self.name!r} must match [A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TrainingPlan:
    """The whole experiment.

    holdout: the card holdout every dojo must have been built with.
    held_out_dojos: registered dojos never placed in any diet; evaluated
        every round so evaluation/ can measure transfer to unseen tasks.
    eval_examples_per_dojo: cap passed as max_examples to each per-round
        TEST pass.
    seed: seeds the diet sampler and any other trainer randomness.
    """

    phases: tuple[Phase, ...]
    holdout: HoldoutSpec
    held_out_dojos: frozenset[str]
    eval_examples_per_dojo: int
    seed: int
    faults: FaultPolicy = FaultPolicy()

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("a plan needs at least one phase")
        # Case-folded: on case-insensitive filesystems "A" and "a" would share
        # a checkpoint directory
        names = [phase.name.lower() for phase in self.phases]
        if len(set(names)) != len(names):
            raise ValueError("phase names must be unique")
        if self.eval_examples_per_dojo < 1:
            raise ValueError("eval_examples_per_dojo must be >= 1")
        for phase in self.phases:
            overlap = self.held_out_dojos.intersection(phase.dojo_names)
            if overlap:
                raise ValueError(
                    f"phase {phase.name!r} trains held-out dojos {sorted(overlap)}"
                )


@dataclass(frozen=True)
class HardwareLimits:
    """Machine-dependent ceilings.

    max_batch_cost: ceiling on the summed cost_of of one batch (becomes
        BatchBudget.max_cost).
    """

    max_batch_cost: int

    def __post_init__(self) -> None:
        if self.max_batch_cost < 1:
            raise ValueError("max_batch_cost must be >= 1")
