"""BestHeadKeeper: remember each dojo head's weights at its best TEST round,
so an extrinsic run scores the best head rather than the last one."""

import math
from typing import Mapping, Sequence

import torch

from src.dojos.dojo import Dojo
from src.training.recording.reports import CheckpointRecord, RoundReport


class BestHeadKeeper:
    """A RunListener that snapshots a dojo's head whenever its TEST loss
    reaches a new low, and can put those snapshots back.

    Why: saturation ends a head's training only patience_rounds rounds
    after its best, and on a frozen encoder a head spends those rounds
    overfitting. Scoring the last head would fold that overshoot, which
    differs per encoder, into the comparison.

    The snapshots are detached copies of dojo.trainable_parameters(), in
    that order, on the head's own device (heads are small). Buffers (e.g.
    batch-norm running statistics) are not kept; today's heads have none.

    Inputs (constructor): dojos (the run's dojos; a name not in a report is
        never snapshotted).
    """

    def __init__(self, dojos: Sequence[Dojo]) -> None:
        """Side effects: none. Exceptions: none."""
        self._dojos = {dojo.name: dojo for dojo in dojos}
        self._best_losses: dict[str, float] = {}
        self._best_rounds: dict[str, int] = {}
        self._snapshots: dict[str, list[torch.Tensor]] = {}

    @property
    def best_rounds(self) -> Mapping[str, int]:
        """Dojo name -> round index of its best TEST loss so far (a copy).
        Side effects: none. Exceptions: none."""
        return dict(self._best_rounds)

    def on_round_end(self, report: RoundReport) -> None:
        """Snapshot every dojo whose normalized TEST loss this round is its
        lowest yet.

        Inputs: report. Output: None.
        Side effects: copies the improved heads' parameters.
        Exceptions: none (a dojo not in the report, or with a non-finite
            loss, is skipped).

        Example:
            >>> keeper.on_round_end(report)  # report.per_dojo_test_loss["a"].normalized = 0.8
            >>> keeper.best_rounds["a"]
            3
        """
        # Normalized loss, as the Trainer's own decisions use (same ranking
        # as raw loss for a dojo whose baseline is one constant)
        for name, split_loss in report.per_dojo_test_loss.items():
            loss = split_loss.normalized
            dojo = self._dojos.get(name)
            # A NaN would otherwise win: nan >= best is always False
            if dojo is None or not math.isfinite(loss):
                continue
            if loss >= self._best_losses.get(name, math.inf):
                continue
            self._best_losses[name] = loss
            self._best_rounds[name] = report.round_index
            self._snapshots[name] = [
                parameter.detach().clone() for parameter in dojo.trainable_parameters()
            ]

    def on_checkpoint(self, record: CheckpointRecord) -> None:
        """Nothing: extrinsic runs have no checkpointer."""

    def restore_best_heads(self) -> None:
        """Copy each snapshotted dojo's best parameters back into its head.
        Dojos never snapshotted keep their current weights.

        Inputs: none. Output: None.
        Side effects: overwrites head parameters in place.
        Exceptions: ValueError if a head's parameter count or shapes changed
            since its snapshot.

        Example:
            >>> keeper.restore_best_heads()
        """
        for name, snapshot in self._snapshots.items():
            parameters = list(self._dojos[name].trainable_parameters())
            if len(parameters) != len(snapshot):
                raise ValueError(f"dojo {name!r} head changed shape since its snapshot")
            with torch.no_grad():
                for parameter, saved in zip(parameters, snapshot):
                    if parameter.shape != saved.shape:
                        raise ValueError(
                            f"dojo {name!r} head changed shape since its snapshot"
                        )
                    parameter.copy_(saved)
