"""Exercise one dojo before it ever reaches Trainer.run().

A dojo built against a stale split file (e.g. gwent_one's own metrics
were once built against a pre-re-ingest CardBinder, so every uuid
failed to resolve) raises nothing on its own -
Trainer just yields zero-example batches and quarantines it a few rounds
in. This runs the same forward/loss path Trainer's _take_step does,
against random embeddings shaped like the dojo's own batch (no encoder,
no GPU, no LM download needed), so that kind of mistake surfaces in
seconds instead of after the first round of an unattended run.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Union

import torch

from src.dojos.dojo import BatchBudget, Dojo, require_usable_baseline
from src.dojos.mods.mod import ModTally
from src.schema.card import GenericCard
from src.schema.padded_embeddings import ModelEmbeddings, PaddedEmbeddings
from src.schema.splits import Split
from src.schema.type_hints import (
    BatchedTrainingInput,
    InputShape,
    TrainingInput,
    batched_input_shape_of,
)


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
    mod_tallies: a snapshot (copies) of the dojo's augmentation-mod
        tallies after the one TRAIN batch, keyed as Dojo.mod_tallies.
        Informational: a mod that never fired or failed does not make the
        dojo fail preflight (mods are best effort by design).
    sample_baseline_loss: dojo.baseline_loss of that same TRAIN batch (the
        "learned nothing" loss normalized loss divides by); None if any
        earlier step failed.
    """

    dojo_name: str
    train_count: int
    test_count: int
    sample_loss: float | None
    error: str | None
    mod_tallies: Mapping[str, ModTally] = field(default_factory=dict)
    sample_baseline_loss: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def random_embeddings_like(
    inputs: BatchedTrainingInput, card_embedding_size: int
) -> ModelEmbeddings:
    """Random embeddings shaped exactly as CardEncoderModel.forward would
    return them for inputs, without running any encoder.

    The one stand-in builder for preflight, scripts/survey_dojo_baselines.py
    and head/dojo tests, so a dojo's compute_loss sees the same types as in
    training.
    Inputs: inputs (a non-empty batch of any shape), card_embedding_size.
    Output: ModelEmbeddings - PaddedEmbeddings (SINGLE_CARD or MULTI_CARD)
        of torch.randn cards, zero at padding; for multi-group inputs, the
        nested lists with one torch.randn(card_embedding_size) per card.
    Side effects: draws from torch's global RNG.
    Exceptions: as batched_input_shape_of for a malformed batch.

    Example:
        >>> random_embeddings_like([[card_a, card_b], [card_c]], 32).lengths
        (2, 1)
    """
    # Classify one example, like the model does
    shape = batched_input_shape_of(inputs)

    # Multi-group stays nested; the padded shapes pack through from_nested
    if shape is InputShape.MULTI_GROUP:
        return _random_nested(inputs, card_embedding_size)
    nested = _random_nested(inputs, card_embedding_size)
    return PaddedEmbeddings.from_nested(nested, shape, card_embedding_size)


def _random_nested(
    inputs: Union[TrainingInput, BatchedTrainingInput], card_embedding_size: int
) -> Any:
    """Random tensors nested exactly like inputs: one
    torch.randn(card_embedding_size) per GenericCard leaf.

    Private helper - single caller is random_embeddings_like. Recursive,
    like iter_cards: the nesting is at most three lists deep.
    Inputs: inputs (a GenericCard or list nesting of them), width.
    Output: the matching nesting of tensors (a Tensor or nested lists of
        them; Any because the depth is only known at runtime, and the
        caller narrows by the shape it classified).
    Side effects: draws from torch's global RNG. Exceptions: none.
    """
    if isinstance(inputs, GenericCard):
        return torch.randn(card_embedding_size)
    return [_random_nested(item, card_embedding_size) for item in inputs]


def preflight_dojo(
    dojo: Dojo, budget: BatchBudget, card_embedding_size: int
) -> PreflightResult:
    """Construct-time exercise of one already-built dojo.

    Inputs: dojo (built against real data), budget (as Trainer would
        build from HardwareLimits), card_embedding_size (the width the
        dojo's own decoder head was built for).
    Output: PreflightResult; .ok is True only if every step below ran
        without error and produced a finite scalar loss.
    Side effects: pulling the TRAIN batch runs the dojo's augmentation
        mods, advancing their random state and tallies (snapshotted into
        the result, on success and failure alike). Nothing calls
        .backward(), so no gradient is ever populated.
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
        embeddings = random_embeddings_like(batch.inputs, card_embedding_size)
        loss = dojo.compute_loss(embeddings, batch)
        if loss.dim() != 0:
            raise ValueError(
                f"compute_loss returned shape {tuple(loss.shape)}, want a scalar"
            )
        if not torch.isfinite(loss):
            raise ValueError(f"compute_loss returned non-finite {loss.item()}")
        # A degenerate baseline would make normalized loss undefined mid-run
        baseline = require_usable_baseline(dojo.baseline_loss(batch), dojo.name)
        return PreflightResult(
            dojo.name,
            train_count,
            test_count,
            loss.item(),
            None,
            _tally_snapshot(dojo),
            sample_baseline_loss=baseline,
        )
    except Exception as error:
        return PreflightResult(
            dojo.name,
            train_count,
            test_count,
            None,
            f"{type(error).__name__}: {error}",
            _tally_snapshot(dojo),
        )


def _tally_snapshot(dojo: Dojo) -> dict[str, ModTally]:
    """Copies of dojo's mod tallies, so later training cannot change what a
    PreflightResult reports.

    Inputs: dojo. Output: dict[str, ModTally]; empty if the dojo cannot
        report its tallies (so preflight_dojo keeps its no-raise promise).
    Side effects: none. Exceptions: none.
    """
    try:
        tallies = dojo.mod_tallies()
    except Exception:  # a report must never turn a check into a crash
        return {}
    return {label: dataclasses.replace(tally) for label, tally in tallies.items()}
