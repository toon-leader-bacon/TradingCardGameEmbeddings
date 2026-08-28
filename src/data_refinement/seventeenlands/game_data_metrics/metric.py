"""The shared interface every per-card numeric metric implements.

See src/data_refinement/README.md for this container's scope and
this container's own README for the full contract this was built
against.

Metric is a Strategy (PATTERNS.md) — one implementation per metric
(e.g. win_rate — see metrics/win_rate/win_rate.py's WinRateMetric),
swappable behind this shared interface so MetricScanner (metric_scanner.py)
can drive any number of active metrics over one streamed pass of a raw
CSV, polymorphically. Implemented as a typing.Protocol (structural
typing — matching the convention already established by
src/data_refinement/card_binder/ingestion.py's CardIngestionStage),
not an ABC.

Each Metric owns its own running accumulator state internally, so
MetricScanner never needs to know a metric's internals — accumulate()
updates that state per chunk, finalize() turns it into MetricResult
rows, and save_state()/load_state() are a Memento pair (PATTERNS.md)
letting MetricScanner checkpoint and resume a metric's accumulator
without violating its encapsulation (see metric_scanner.py for why
checkpointing exists at all — a full pass over a multi-GB CSV can run
long enough that surviving a crash/restart matters).
"""

from typing import Protocol
from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.metric_result import MetricResult


class Metric(Protocol):
    """Strategy: accumulate one numeric metric across chunks of game_data.

    Implementations must be safe to call accumulate() many times in a
    row (once per chunk) before finalize() is ever called — finalize()
    itself must be safe to call at most once per meaningful result (no
    requirement to support calling it twice and getting the same
    answer, since MetricScanner only ever calls it once, at the end of
    a scan). Single-consumer to MetricScanner, which is agnostic to
    which concrete implementation it's driving.
    """

    name: str

    def accumulate(
        self, chunk: pd.DataFrame, card_columns: list[CardColumnSet]
    ) -> None:
        """Update this metric's running accumulator with one chunk of rows.

        Implementations should use vectorized pandas operations across
        each CardColumnSet's columns (e.g. boolean masks summed via
        chunk[col].sum()), not a per-row Python loop — the number of
        rows in a chunk can be large, but the number of distinct cards
        (len(card_columns)) is small and fixed, so a Python-level loop
        over card_columns (each iteration doing vectorized work across
        the whole chunk) is the intended shape, not a smell.

        Inputs:
            chunk: one chunk of the raw game_data CSV, as read by
                pandas.read_csv(..., chunksize=...) — includes both
                metadata columns (e.g. "won") and every card's five
                columns.
            card_columns: every successfully resolved card's column
                names for this file (see column_lookup.py) — the
                same list, unchanged, on every call within one
                MetricScanner scan.
        Output: none.
        Side effects: mutates this metric's internal accumulator state.
        Exceptions: none expected from well-formed input.
        """
        ...

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn this metric's accumulated state into final MetricResult rows.

        Inputs: none (uses this metric's internal accumulator state,
            built up across prior accumulate() calls).
        Output: one MetricResult per nocab_uuid this metric has seen
            enough data to produce a result for. No minimum-sample-size
            cutoff is applied here — every card with at least one
            observation gets a result, with sample_size reported
            honestly so a downstream consumer (a dojo) decides its own
            noise cutoff.
        Side effects: none — does not reset or mutate accumulator
            state (MetricScanner calls this once, at the very end of a
            scan).
        Exceptions: none expected.
        """
        ...

    def save_state(self) -> dict:
        """Snapshot this metric's accumulator state as serializable data.

        Memento (PATTERNS.md) — the returned dict must be JSON-
        serializable (MetricScanner writes checkpoints as JSON), which
        means any UUID-keyed internal state must be stringified here,
        not left as UUID objects.

        Inputs: none (uses this metric's internal accumulator state).
        Output: a JSON-serializable dict capturing enough state to
            fully restore this metric via load_state() later.
        Side effects: none.
        Exceptions: none expected.
        """
        ...

    def load_state(self, state: dict) -> None:
        """Restore this metric's accumulator state from a save_state() snapshot.

        Memento (PATTERNS.md) counterpart to save_state() — called by
        MetricScanner before resuming a streamed pass from a
        checkpoint, on a freshly constructed metric instance (not one
        that's already accumulated anything).

        Inputs:
            state: a dict previously returned by this metric's own
                save_state().
        Output: none.
        Side effects: overwrites this metric's internal accumulator
            state.
        Exceptions: raises if state isn't shaped as this metric's own
            save_state() would have produced.
        """
        ...
