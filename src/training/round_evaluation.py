"""A scoring pass: every dojo's mean loss on one split, without training.

Trainer calls it each round on TEST, to feed the SaturationTracker (the
loop's own cheap signal). evaluation/'s extrinsic runs call it once on
VALIDATION after training finishes. It never calls evaluation/.
"""

import logging
from typing import Mapping, Sequence

import torch

from src.dojos.dojo import BatchBudget, Dojo, require_usable_baseline
from src.encoder_model.precision import Precision, inference_context
from src.schema.splits import Split
from src.training.recording.reports import SplitLoss
from src.training.trainable_encoder import TrainableEncoder

logger = logging.getLogger(__name__)


def evaluate_split_losses(
    model: TrainableEncoder,
    dojos: Sequence[Dojo],
    *,
    split: Split,
    budget: BatchBudget,
    max_examples: int,
    precision: Precision,
) -> Mapping[str, SplitLoss]:
    """Mean loss and baseline of every dojo on a capped, deterministic subsample of `split`.

    Inputs: model, dojos (every dojo to score), split (the rows to score),
        budget, max_examples (per-dojo cap passed to Dojo.batches), and
        precision (forward passes run under inference_context(model,
        precision): eval mode, no_grad, autocast).
    Output: dojo.name -> SplitLoss: per-example mean loss and mean
        baseline_loss, weighting each batch by len(batch) (so .normalized
        compares the dojo with its own "learned nothing" loss). A dojo
        whose pass fails is logged and omitted.
    Side effects: none on parameters (eval mode, torch.no_grad); the
        model's previous train/eval mode is restored even on error.
    Exceptions: none for a single dojo's failure (logged, dojo omitted).
        Entering autocast itself can raise (a dtype the device does not
        support); that is not per-dojo and propagates.

    Example:
        >>> evaluate_split_losses(model, dojos, split=Split.TEST, budget=budget,
        ...                       max_examples=512, precision="fp16")["pick"]
        SplitLoss(loss=1.37, baseline_loss=2.71)
    """
    result: dict[str, SplitLoss] = {}
    # Score each dojo in eval mode, without autograd graphs, at `precision`
    with inference_context(model, precision):
        for dojo in dojos:
            try:
                result[dojo.name] = _split_loss(
                    model, dojo, split, budget, max_examples
                )
            except Exception:
                logger.warning(
                    "%s pass failed for %s", split.value, dojo.name, exc_info=True
                )
    return result


def _split_loss(
    model: TrainableEncoder,
    dojo: Dojo,
    split: Split,
    budget: BatchBudget,
    max_examples: int,
) -> SplitLoss:
    """Example-weighted mean loss and baseline over one dojo's capped pass
    of `split`.

    Per batch b of n_b examples: total_loss += loss_b * n_b and
    total_baseline += dojo.baseline_loss(b) * n_b, so SplitLoss.normalized
    is a ratio of example-weighted means (for a generic dojo, whose
    baseline is one constant, exactly mean loss / baseline).

    Inputs: model, dojo, split, budget, max_examples (see
        evaluate_split_losses).
    Output: SplitLoss(total_loss / n, total_baseline / n).
    Side effects: forward passes (the caller holds eval mode / no_grad).
    Exceptions: ValueError if the dojo yields no batches of `split`, a loss
        is non-finite, or a baseline is not finite and > 0 (caught per dojo
        by evaluate_split_losses); TypeError from baseline_loss for a
        foreign batch.
    """
    total_loss = 0.0
    total_baseline = 0.0
    total_examples = 0
    for batch in dojo.batches(split, budget, max_examples):
        loss = dojo.compute_loss(model(batch.inputs), batch)
        if not torch.isfinite(loss).all():
            raise ValueError(f"non-finite {split.value} loss for dojo {dojo.name!r}")
        baseline = require_usable_baseline(
            dojo.baseline_loss(batch), f"dojo {dojo.name!r}"
        )
        total_loss += loss.item() * len(batch)
        total_baseline += baseline * len(batch)
        total_examples += len(batch)
    if total_examples == 0:
        raise ValueError(f"dojo {dojo.name!r} yielded no {split.value} examples")
    return SplitLoss(
        loss=total_loss / total_examples,
        baseline_loss=total_baseline / total_examples,
    )
