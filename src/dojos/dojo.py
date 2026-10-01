"""The one public interface the training loop drives every dojo through.

Generic (per-example label) and contrastive (deck-pool) dojos differ in
their batch type and loss, but the trainer needs neither detail: it asks
for batches under a BatchBudget, runs the encoder on `batch.inputs`, and
hands the embeddings back to `compute_loss`. `baseline_loss` puts every
dojo's loss on one scale for decisions (see src/dojos/README.md).
"""

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, Protocol

import torch
from torch import nn

from src.dojos.mods.mod import ModTally
from src.schema.card import GenericCard
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.schema.type_hints import BatchedModelOutput, BatchedTrainingInput


@dataclass(frozen=True)
class BatchBudget:
    """How much one batch may cost, set by the trainer from hardware limits.

    max_cost: ceiling on the summed cost_of over every card in a batch.
    cost_of: cost of one card; 1 per card by default at the trainer,
        upgradable to serialized length or token count without touching
        any dojo.
    """

    max_cost: int
    cost_of: Callable[[GenericCard], int]


def require_usable_baseline(baseline_loss: float, context: str) -> float:
    """baseline_loss, checked against Dojo.baseline_loss's contract.

    Normalized loss is loss / baseline_loss, so a baseline must be finite
    and > 0; a zero baseline (e.g. one class only) leaves it undefined.
    Inputs: baseline_loss (float), context (str naming its source, for the
        error message).
    Output: baseline_loss, unchanged.
    Side effects: none.
    Exceptions: ValueError naming context if baseline_loss is not finite
        or is <= 0.

    Example:
        >>> require_usable_baseline(0.69, "pick")
        0.69
    """
    if not math.isfinite(baseline_loss) or baseline_loss <= 0.0:
        raise ValueError(
            f"{context}: baseline_loss must be finite and > 0, got {baseline_loss!r}"
        )
    return baseline_loss


class DojoBatch(Protocol):
    """One yielded batch: encoder inputs plus whatever the dojo's loss needs."""

    inputs: BatchedTrainingInput

    def __len__(self) -> int:
        """Example count (not card count)."""
        ...


class Dojo(Protocol):
    """A pluggable training task the trainer can sample from and score."""

    name: str
    holdout: HoldoutSpec

    def batches(
        self, split: Split, budget: BatchBudget, max_examples: int | None = None
    ) -> Iterator[DojoBatch]:
        """Yield batches of one split.

        Every batch's summed budget.cost_of over its cards is at most
        budget.max_cost. max_examples truncates to the first N examples of
        the split in its fixed file order, so a capped TEST pass is
        deterministic.
        """
        ...

    def example_count(self, split: Split) -> int:
        """Approximate number of examples in a split (counted before skips)."""
        ...

    def compute_loss(
        self, embeddings: BatchedModelOutput, batch: DojoBatch
    ) -> torch.Tensor:
        """Scalar per-example MEAN loss for the batch's embeddings."""
        ...

    def baseline_loss(self, batch: DojoBatch) -> float:
        """The per-example mean loss an input-ignoring predictor gets on
        batch: the "learned nothing" reference for normalized loss
        (loss / baseline; 1.0 = learned nothing, 0 = perfect).

        Depends on the dojo's data only, never on the encoder or head
        weights. A dojo may return one constant fit to its TRAIN split
        (generic dojos), or compute it from the batch's own shape
        (contrastive dojos, whose baseline depends on how many items and
        cliques a batch holds).
        Inputs: batch, one this dojo yielded.
        Output: float, finite and > 0.
        Side effects: none.
        Exceptions: TypeError if batch is not this dojo's batch type;
            ValueError if the batch has no usable baseline (e.g. no
            negatives at all).
        """
        ...

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """The dojo's own (decoder head) parameters, excluding the encoder."""
        ...

    def mod_tallies(self) -> Mapping[str, ModTally]:
        """Live tallies of this dojo's augmentation mods, keyed by a label
        naming each mod (e.g. "1:WeightedFieldMaskMod"); empty when the
        dojo has none. For reports only: callers must not change them.
        Side effects: none. Exceptions: none.
        """
        ...

    def move_head_to(self, device: torch.device) -> None:
        """Move the head's parameters and buffers to device, in place.

        The trainer calls this so heads sit beside the encoder's output.
        Side effects: moves the head. Exceptions: whatever torch raises
        for an unavailable device.
        """
        ...

    def reset_head(self) -> None:
        """Restore the head to its as-built initial state.

        Deterministic: repeated calls on one dojo give the same head, so
        runs that each start with reset_head() start from identical heads
        (e.g. evaluation's extrinsic runs across several encoders).
        Side effects: overwrites the head's weights. Exceptions: none.
        """
        ...
