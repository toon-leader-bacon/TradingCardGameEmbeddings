"""An endless TRAIN batch source for one dojo."""

from typing import Iterator

from src.dojos.dojo import BatchBudget, Dojo, DojoBatch
from src.schema.splits import Split


class DojoBatchStream:
    """Hands out TRAIN batches one at a time, starting a fresh pass over
    the split whenever the previous one is exhausted.

    Inputs (constructor): dojo (Dojo), budget (BatchBudget).
    """

    def __init__(self, dojo: Dojo, budget: BatchBudget) -> None:
        self._dojo = dojo
        self._budget = budget
        self._batches: Iterator[DojoBatch] | None = None

    def next_batch(self) -> DojoBatch:
        """Return the next TRAIN batch.

        Inputs: none.
        Output: DojoBatch.
        Side effects: advances (or restarts) the dojo's TRAIN iterator.
        Exceptions: RuntimeError if the dojo yields no batches at all; any
            error the dojo raises mid-pass (the next call starts a fresh
            pass, so the fault ledger's total-failure limit is what stops a
            dojo that fails at the same row every pass).

        Example:
            >>> DojoBatchStream(dojo, budget).next_batch()
        """
        # Start a pass if none is open
        if self._batches is None:
            self._batches = self._dojo.batches(Split.TRAIN, self._budget)
        # Take the next batch; StopIteration means the pass ended, but any
        # other error means the dead generator must not be mistaken for that
        try:
            return next(self._batches)
        except StopIteration:
            pass
        except Exception:
            self._batches = None  # the next call starts a fresh pass
            raise
        self._batches = self._dojo.batches(Split.TRAIN, self._budget)
        batch = next(self._batches, None)
        if batch is None:
            raise RuntimeError(f"dojo {self._dojo.name!r} yielded no TRAIN batches")
        return batch
