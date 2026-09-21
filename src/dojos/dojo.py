"""The one public interface the training loop drives every dojo through.

Generic (per-example label) and contrastive (deck-pool) dojos differ in
their batch type and loss, but the trainer needs neither detail: it asks
for batches under a BatchBudget, runs the encoder on `batch.inputs`, and
hands the embeddings back to `compute_loss`.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Protocol

import torch
from torch import nn

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

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """The dojo's own (decoder head) parameters, excluding the encoder."""
        ...

    def reset_head(self) -> None:
        """Re-initialize the dojo's head, e.g. to measure fresh-head fit speed."""
        ...
