"""Tiny stand-ins for a dojo and an encoder so the loop runs in milliseconds."""

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import nn

from src.dojos.dojo import BatchBudget, DojoBatch
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.training.plan import (
    HardwareLimits,
    Phase,
    Proportional,
    SaturationSpec,
    TrainingPlan,
)

BUDGET = BatchBudget(max_cost=10, cost_of=lambda card: 1)
LIMITS = HardwareLimits(max_batch_cost=10)


@dataclass
class FakeBatch:
    inputs: Any
    size: int = 4

    def __len__(self) -> int:
        return self.size


class FakeModel(nn.Module):
    """Linear encoder; satisfies TrainableEncoder."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 2)

    def forward(self, inputs: Any) -> Any:
        return self.layer(inputs)

    def encoder_only_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.state_dict()


class FakeDojo:
    """Regression-to-zero dojo with a linear head.

    fail: 'train' raises when TRAIN batches are requested, 'test' when
    TEST ones are; nan_loss makes compute_loss return NaN.
    """

    def __init__(
        self,
        name: str,
        n_batches: int = 3,
        train_count: int = 100,
        fail: str | None = None,
        nan_loss: bool = False,
        with_head: bool = True,
        count_fails: bool = False,
        nan_grad: bool = False,
        fail_at_batch: int | None = None,
    ) -> None:
        self.name = name
        self.holdout = HoldoutSpec.no_holdout()
        self._n_batches = n_batches
        self._train_count = train_count
        self._fail = fail
        self._nan_loss = nan_loss
        self._head = nn.Linear(2, 1)
        self._with_head = with_head
        self._count_fails = count_fails
        self._nan_grad = nan_grad
        self._fail_at_batch = fail_at_batch

    def batches(
        self, split: Split, budget: BatchBudget, max_examples: int | None = None
    ) -> Iterator[DojoBatch]:
        if self._fail == split.value:
            raise RuntimeError(f"{self.name} cannot serve {split}")
        for index in range(self._n_batches):
            if split is Split.TRAIN and index == self._fail_at_batch:
                raise RuntimeError(f"{self.name} bad row in batch {index}")
            yield FakeBatch(torch.randn(4, 3))

    def example_count(self, split: Split) -> int:
        if self._count_fails:
            raise OSError("cannot read split file")
        return self._train_count

    def compute_loss(self, embeddings: Any, batch: Any) -> torch.Tensor:
        loss = self._head(embeddings).pow(2).mean()
        if self._nan_grad:
            # finite forward value, NaN gradient (sqrt at 0)
            loss = loss + (embeddings.sum() * 0).sqrt()
        return loss * float("nan") if self._nan_loss else loss

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self._head.parameters() if self._with_head else []

    def reset_head(self) -> None:
        pass


def saturation_spec(**overrides: Any) -> SaturationSpec:
    fields: dict[str, Any] = dict(
        epsilon=0.0,
        patience_rounds=2,
        reactivation_delta=1.0,
        target_saturated_fraction=1.0,
    )
    fields.update(overrides)
    return SaturationSpec(**fields)


def make_phase(dojo_names: tuple[str, ...], **overrides: Any) -> Phase:
    fields: dict[str, Any] = dict(
        name="joint",
        dojo_names=dojo_names,
        diet_rule=Proportional(),
        encoder_trainable=True,
        encoder_lr=0.01,
        head_lr=0.01,
        steps_per_round=4,
        max_rounds=3,
        saturation=saturation_spec(),
    )
    fields.update(overrides)
    return Phase(**fields)


def make_plan(phases: tuple[Phase, ...], **overrides: Any) -> TrainingPlan:
    fields: dict[str, Any] = dict(
        phases=phases,
        holdout=HoldoutSpec.no_holdout(),
        held_out_dojos=frozenset(),
        eval_examples_per_dojo=8,
        seed=0,
    )
    fields.update(overrides)
    return TrainingPlan(**fields)
