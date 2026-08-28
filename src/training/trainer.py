"""The shared Trainer contract every training loop implementation exposes.

See plans/training_pipeline.md. Implemented as an ABC rather than a
Protocol (unlike Dojo/CardIngestionStage) since every Trainer shares
real constructor plumbing (a model, a dojo, a corpus) worth factoring
up, not just a shared method shape.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EvaluationResult:
    """The outcome of one Trainer.train() call's test/validate phases.

    test_loss: the diagnostic-only score from Split.TEST, run once
        after training finishes (see plans/training_pipeline.md — not
        currently acted on, reported for visibility only).
    validate_loss: this run's actual reported result, from Split.VALIDATE,
        touched exactly once, never during the train or test phases.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    test_loss: torch.Tensor
    validate_loss: torch.Tensor


class Trainer(ABC):
    """Runs an encoder + dojo(s) through optimizer steps to a checkpoint.

    One implementation per model-arity/regime combination —
    SingleCardTrainer (single_card_trainer.py) is the first. See
    plans/training_pipeline.md for why this is a flat family of
    sibling implementations rather than one parameterized class.
    """

    @abstractmethod
    def train(self, num_steps: int, batch_size: int) -> EvaluationResult:
        """Run num_steps train-split optimizer steps, then score test/validate.

        Three sequential phases (plans/training_pipeline.md): num_steps
        gradient steps against Split.TRAIN, then one full no-gradient
        pass over Split.TEST, then one full no-gradient pass over
        Split.VALIDATE.

        Inputs:
            num_steps: how many train-split optimizer steps to run.
            batch_size: how many examples per step/pass batch.
        Output: this run's EvaluationResult.
        Side effects: mutates the encoder's and any decoder heads'
            weights; advances the optimizer's internal state.
        Exceptions: implementation-defined.
        """
        raise NotImplementedError
