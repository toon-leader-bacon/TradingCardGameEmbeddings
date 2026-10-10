"""Experiment definition: what to train, in which phases, on which cards.

Everything here is a frozen, hardware-independent value so a run can be
serialized into its manifest and replayed. Hardware limits live apart in
HardwareLimits because they change per machine, not per experiment (they
affect results but are not yet written to the checkpoint manifest).
"""

import math
import re
from dataclasses import dataclass
from typing import get_args

from src.encoder_model.precision import Precision
from src.schema.holdout import HoldoutSpec
from src.training.loss_weighting import LossWeighting


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


# The rules that weigh a flat list of dojos by TRAIN count alone
FlatDietRule = Proportional | Uniform | Temperature


@dataclass(frozen=True)
class DojoRow:
    """A diet table row naming one dojo.

    weight: the row's relative chance among its siblings (finite, >= 0).
    dojo_name: a run dojo, not held out.
    """

    weight: float
    dojo_name: str

    def __post_init__(self) -> None:
        _check_row_weight(self.weight)


@dataclass(frozen=True)
class DojoGroupRow:
    """A diet table row over several dojos, weighted among themselves by a
    flat rule (a config `dojos:` row: its name patterns already matched
    against the run's dojos).

    weight: the whole group's relative chance among its siblings.
    dojo_names: the matched dojos, in run order, at least one.
    within: how the group splits its share (by TRAIN count ** alpha).
    """

    weight: float
    dojo_names: tuple[str, ...]
    within: FlatDietRule

    def __post_init__(self) -> None:
        _check_row_weight(self.weight)
        if not self.dojo_names:
            raise ValueError("a dojo group row needs at least one dojo")


@dataclass(frozen=True)
class SubTableRow:
    """A diet table row that rolls again on its own rows.

    weight: the sub-table's relative chance among its siblings.
    rows: the sub-table, at least one row with a positive weight.
    """

    weight: float
    rows: tuple["DietTableRow", ...]

    def __post_init__(self) -> None:
        _check_row_weight(self.weight)
        _check_rows_pullable(self.rows)


DietTableRow = DojoRow | DojoGroupRow | SubTableRow


@dataclass(frozen=True)
class TableDiet:
    """Sample dojos from a nested drop table (src/utils/drop_table.py).

    Saturated and quarantined dojos drop out of their own sub-table, so
    their share goes to their siblings; a row left with no active dojo
    drops out and its share goes to the rows beside it.

    rows: the top-level table, at least one row with a positive weight.
        No dojo appears under two rows.
    """

    rows: tuple[DietTableRow, ...]

    def __post_init__(self) -> None:
        _check_rows_pullable(self.rows)
        duplicates = sorted(
            {name for name in self.dojo_names if self.dojo_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"diet table lists {duplicates} more than once")

    @property
    def dojo_names(self) -> tuple[str, ...]:
        """Every dojo under the table, in row order (depth first).

        Inputs: none. Output: tuple[str, ...]. Side effects: none.
        Exceptions: none.

        Example:
            >>> TableDiet((DojoRow(1, "a"), DojoRow(1, "b"))).dojo_names
            ('a', 'b')
        """
        return leaf_dojo_names(self.rows)


DietRule = FlatDietRule | TableDiet


def _check_row_weight(weight: float) -> None:
    """Raise unless weight is finite and >= 0.

    Inputs: weight. Output: none. Side effects: none.
    Exceptions: ValueError (also for NaN).
    """
    # `not x >= 0` also rejects NaN
    if not (math.isfinite(weight) and weight >= 0):
        raise ValueError(f"row weight must be finite and >= 0, got {weight}")


def _check_rows_pullable(rows: tuple[DietTableRow, ...]) -> None:
    """Raise unless rows is non-empty with at least one positive weight
    (the same rule DropTable enforces, checked at parse time so a bad table
    fails before any dojo is built).

    Inputs: rows. Output: none. Side effects: none.
    Exceptions: ValueError.
    """
    if not rows:
        raise ValueError("a diet table needs at least one row")
    if not any(row.weight > 0 for row in rows):
        raise ValueError("a diet table needs at least one positive weight")


def leaf_dojo_names(rows: tuple[DietTableRow, ...]) -> tuple[str, ...]:
    """Every dojo under rows, depth first in row order. A loop with an
    explicit stack of rows still to visit, not recursion.

    Inputs: rows. Output: tuple[str, ...] (duplicates kept, so the caller
    can report them). Side effects: none. Exceptions: none.

    Example:
        >>> leaf_dojo_names((DojoRow(1, "a"), SubTableRow(1, (DojoRow(1, "b"),))))
        ('a', 'b')
    """
    result: list[str] = []
    # Reversed onto the stack so rows pop in their written order
    stack: list[DietTableRow] = list(reversed(rows))

    # Visit each row; a sub-table pushes its own rows to visit next
    while stack:
        row = stack.pop()
        if isinstance(row, DojoRow):
            result.append(row.dojo_name)
        elif isinstance(row, DojoGroupRow):
            result.extend(row.dojo_names)
        else:
            stack.extend(reversed(row.rows))
    return tuple(result)


@dataclass(frozen=True)
class SaturationSpec:
    """When a dojo stops being worth training on.

    All loss thresholds apply to NORMALIZED TEST loss (loss / the dojo's
    baseline loss), so they mean the same for every dojo: 0.001 is 0.1% of
    that dojo's "learned nothing" loss.

    epsilon: a round counts as an improvement only if normalized TEST loss
        drops by more than this below the dojo's best so far.
    patience_rounds: consecutive non-improving rounds before SATURATED.
    reactivation_delta: a SATURATED dojo re-enters the diet if its TEST
        loss rises more than this above its loss when it saturated.
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
        With a TableDiet, exactly the table's dojos.
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
        # A table diet names its own dojos: the phase's list must be the same
        if isinstance(self.diet_rule, TableDiet) and set(self.dojo_names) != set(
            self.diet_rule.dojo_names
        ):
            raise ValueError(
                f"phase {self.name!r} dojos differ from its diet table's dojos"
            )


@dataclass(frozen=True)
class TrainingPlan:
    """The whole experiment.

    holdout: the card holdout every dojo must have been built with.
    held_out_dojos: registered dojos never placed in any diet, scored
        every round with never-trained heads: the loop's own signal of
        transfer. evaluation/'s extrinsic runs instead train fresh heads
        for such dojos on the frozen encoder.
    eval_examples_per_dojo: cap passed as max_examples to each per-round
        TEST pass.
    seed: seeds the diet sampler and any other trainer randomness.
    faults: how failed steps are tolerated.
    loss_weighting: per-dojo weights and the baseline floor applied to every
        training step's loss (src/training/loss_weighting.py). None trains
        on the raw loss, as the extrinsic runs do. Every weight must name a
        dojo in some phase's diet.
    """

    phases: tuple[Phase, ...]
    holdout: HoldoutSpec
    held_out_dojos: frozenset[str]
    eval_examples_per_dojo: int
    seed: int
    faults: FaultPolicy = FaultPolicy()
    loss_weighting: LossWeighting | None = None

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
        # Every weighted dojo must be trained in some phase
        if self.loss_weighting is not None:
            trained = {name for phase in self.phases for name in phase.dojo_names}
            untrained = sorted(set(self.loss_weighting.weights) - trained)
            if untrained:
                raise ValueError(f"loss_weights names {untrained}, in no phase's diet")


@dataclass(frozen=True)
class HardwareLimits:
    """Machine-dependent ceilings and numeric format.

    max_batch_cost: ceiling on the summed cost_of of one batch (becomes
        BatchBudget.max_cost).
    precision: "fp32" runs everything in float32. "fp16" and "bf16" run
        the forward pass (training and evaluation) under torch.autocast on
        the model's device; weights and optimizer state stay float32.
        "fp16" also scales the loss with a GradScaler so small gradients
        do not underflow; "bf16" has float32's range and needs none. Pick
        by hardware: the RX 6800 runs fp16 about 3x faster than bf16.
    """

    max_batch_cost: int
    precision: Precision = "fp32"

    def __post_init__(self) -> None:
        if self.max_batch_cost < 1:
            raise ValueError("max_batch_cost must be >= 1")
        if self.precision not in get_args(Precision):
            raise ValueError(f"precision must be one of {get_args(Precision)}")
