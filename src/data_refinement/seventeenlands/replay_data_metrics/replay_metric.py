"""The shared interface every per-card replay_data metric implements.

See src/data_refinement/README.md for this container's scope and
game_data_metrics/metric.py's Metric for the sibling interface
this mirrors — same Strategy (PATTERNS.md) shape (accumulate() per
chunk, finalize() once, save_state()/load_state() as a Memento pair for
checkpointing), same typing.Protocol convention (structural, not
inherited), deliberately NOT reusing Metric itself: Metric.
accumulate() takes a `card_columns: list[CardColumnSet]` second
argument (game_data resolves card identity from the CSV header, once);
replay_data has no per-card columns to resolve — card identity is
per-ROW, discovered while streaming (see arena_id_cache.py) — so a
ReplayMetric's accumulate() takes a `resolved: pd.Series` second
argument instead, matching draft_data_metrics' own
`accumulate(chunk, resolved_picks: pd.Series)` shape more closely than
game_data_metrics' (both containers' card identity lives in row data,
not the header).
"""

from typing import Protocol
from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult


class ReplayMetric(Protocol):
    """Strategy: accumulate one numeric metric across chunks of replay_data.

    Implementations must be safe to call accumulate() many times in a
    row (once per chunk) before finalize() is ever called — finalize()
    itself is only ever called once, at the end of a scan. Single-
    consumer to ReplayMetricScanner, which is agnostic to which
    concrete implementation it's driving.
    """

    name: str

    def accumulate(self, chunk: pd.DataFrame, resolved: pd.Series) -> None:
        """Update this metric's running accumulator with one chunk of rows.

        Inputs:
            chunk: one chunk of the raw replay_data CSV, as read by
                pandas.read_csv(..., chunksize=...) — unmodified raw
                columns only (e.g. "won"). No resolved hand data is
                ever mutated into it (see
                arena_id_cache.ResolvedHands' docstring for why).
            resolved: a pd.Series of arena_id_cache.ResolvedHands,
                index-aligned with chunk, one per row — built once per
                chunk by ReplayMetricScanner via ArenaIdCache.get_hands()
                and handed to every active metric identically, so N
                active metrics still cost one resolution pass, not N.
                Which ResolvedHands field (or which raw chunk column) a
                given implementation actually reads is that
                implementation's own concern (see
                metrics/binary_trigger_rate/base.py).
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
            honestly so a downstream consumer (e.g. a dojo) decides its
            own noise cutoff.
        Side effects: none — does not reset or mutate accumulator
            state (ReplayMetricScanner calls this once, at the very
            end of a scan).
        Exceptions: none expected.
        """
        ...

    def save_state(self) -> dict:
        """Snapshot this metric's accumulator state as serializable data.

        Memento (PATTERNS.md) — the returned dict must be JSON-
        serializable (ReplayMetricScanner writes checkpoints as JSON),
        which means any UUID-keyed internal state must be stringified
        here, not left as UUID objects.

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
        ReplayMetricScanner before resuming a streamed pass from a
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
