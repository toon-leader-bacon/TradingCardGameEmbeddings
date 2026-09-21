"""The per-round TEST pass that feeds the SaturationTracker.

This is the loop's own cheap signal. evaluation/ is separate: it inspects
finished checkpoints and is never called from here.
"""

import logging
from typing import Mapping, Sequence

import torch

from src.dojos.dojo import BatchBudget, Dojo
from src.schema.splits import Split
from src.training.trainable_encoder import TrainableEncoder

logger = logging.getLogger(__name__)


def evaluate_test_losses(
    model: TrainableEncoder,
    dojos: Sequence[Dojo],
    budget: BatchBudget,
    max_examples: int,
) -> Mapping[str, float]:
    """Mean TEST loss of every dojo on a capped, deterministic subsample.

    Inputs: model, dojos (every registered dojo, diet or not), budget, and
        max_examples (per-dojo cap passed to Dojo.batches).
    Output: dojo.name -> per-example mean loss, weighting each batch by
        len(batch). A dojo whose pass fails is logged and omitted.
    Side effects: none on parameters (eval mode, torch.no_grad); the
        model's previous train/eval mode is restored even on error.
    Exceptions: none for a single dojo's failure (logged, dojo omitted).

    Example:
        >>> evaluate_test_losses(model, dojos, budget, 512)["pick"]
        1.37
    """
    result: dict[str, float] = {}
    was_training = model.training
    model.train(False)
    try:
        # Score each dojo without building autograd graphs
        with torch.no_grad():
            for dojo in dojos:
                try:
                    result[dojo.name] = _mean_test_loss(
                        model, dojo, budget, max_examples
                    )
                except Exception:
                    logger.warning("test pass failed for %s", dojo.name, exc_info=True)
    finally:
        model.train(was_training)
    return result


def _mean_test_loss(
    model: TrainableEncoder, dojo: Dojo, budget: BatchBudget, max_examples: int
) -> float:
    """Example-weighted mean loss over one dojo's capped TEST pass.

    Raises ValueError if the dojo yields no TEST batches or a loss is
    non-finite (caught per dojo by evaluate_test_losses).
    """
    total_loss = 0.0
    total_examples = 0
    for batch in dojo.batches(Split.TEST, budget, max_examples):
        loss = dojo.compute_loss(model(batch.inputs), batch)
        if not torch.isfinite(loss).all():
            raise ValueError(f"non-finite TEST loss for dojo {dojo.name!r}")
        total_loss += loss.item() * len(batch)
        total_examples += len(batch)
    if total_examples == 0:
        raise ValueError(f"dojo {dojo.name!r} yielded no TEST examples")
    return total_loss / total_examples
