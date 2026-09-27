"""Exercise one dojo before it ever reaches Trainer.run().

A dojo built against a stale split file (see training/TODO.md section C:
gwent_one's own metrics were once built against a pre-re-ingest
CardBinder, so every uuid failed to resolve) raises nothing on its own -
Trainer just yields zero-example batches and quarantines it a few rounds
in. This runs the same forward/loss path Trainer's _take_step does,
against random embeddings shaped like the dojo's own batch (no encoder,
no GPU, no LM download needed), so that kind of mistake surfaces in
seconds instead of after the first round of an unattended run.
"""

from dataclasses import dataclass
from typing import Any

import torch

from src.dojos.dojo import BatchBudget, Dojo
from src.schema.card import GenericCard
from src.schema.splits import Split


@dataclass(frozen=True)
class PreflightResult:
    """One dojo's health check.

    dojo_name: identity, for a report table.
    train_count / test_count: dojo.example_count() per split; -1 if that
        call itself raised (never reached, or raised before either did).
    sample_loss: compute_loss on one TRAIN batch against random
        embeddings; None if any earlier step failed.
    error: str(exception) from the first failing step, or None if every
        step below succeeded.
    """

    dojo_name: str
    train_count: int
    test_count: int
    sample_loss: float | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def _random_embeddings_like(inputs: Any, card_embedding_size: int) -> Any:
    """Random tensors nested exactly like `inputs`, replacing each
    GenericCard leaf with one torch.randn(card_embedding_size).

    This is the same one-embedding-per-card shape SingleCardModel/
    MultiCardModel produce for a real forward pass (single card -> a
    tensor, multi-card/multi-group -> the matching nested list) - see
    src/schema/type_hints.py's shape taxonomy - so a dojo's compute_loss
    sees embeddings shaped the way it always does, without running any
    encoder at all.
    """
    if isinstance(inputs, GenericCard):
        return torch.randn(card_embedding_size)
    return [_random_embeddings_like(item, card_embedding_size) for item in inputs]


def preflight_dojo(
    dojo: Dojo, budget: BatchBudget, card_embedding_size: int
) -> PreflightResult:
    """Construct-time exercise of one already-built dojo.

    Inputs: dojo (built against real data), budget (as Trainer would
        build from HardwareLimits), card_embedding_size (the width the
        dojo's own decoder head was built for).
    Output: PreflightResult; .ok is True only if every step below ran
        without error and produced a finite scalar loss.
    Side effects: none - nothing calls .backward(), so no gradient is
        ever populated on the dojo's head or anywhere else.
    Exceptions: none; every failure is caught and reported in .error.

    Example:
        >>> preflight_dojo(ColorMaskDojo(binder, holdout, 32), budget, 32).ok
        True
    """
    train_count = test_count = -1
    try:
        train_count = dojo.example_count(Split.TRAIN)
        test_count = dojo.example_count(Split.TEST)
        if train_count <= 0:
            raise ValueError(
                f"{train_count} TRAIN examples (stale split file, or every "
                "uuid failed to resolve against the current CardBinder)"
            )
        try:
            batch = next(dojo.batches(Split.TRAIN, budget))
        except StopIteration:
            raise ValueError(
                f"example_count reports {train_count} TRAIN examples but "
                "batches() yielded none"
            ) from None
        embeddings = _random_embeddings_like(batch.inputs, card_embedding_size)
        loss = dojo.compute_loss(embeddings, batch)
        if loss.dim() != 0:
            raise ValueError(
                f"compute_loss returned shape {tuple(loss.shape)}, want a scalar"
            )
        if not torch.isfinite(loss):
            raise ValueError(f"compute_loss returned non-finite {loss.item()}")
        return PreflightResult(dojo.name, train_count, test_count, loss.item(), None)
    except Exception as error:
        return PreflightResult(
            dojo.name, train_count, test_count, None, f"{type(error).__name__}: {error}"
        )
