"""The shared interface every per-card draft_data metric implements.

See src/data_refinement/README.md for this container's scope. Mirrors
game_data_metrics/metric.py's Metric — same Strategy role, same
Memento checkpoint pair — but with a genuinely different accumulate()
signature, not a copy of Metric's. Metric's
accumulate(chunk, card_columns: list[CardColumnSet]) makes sense
because game_data's card references live in the CSV's HEADER (a fixed
set of `deck_<name>`-style columns, resolved once up front and reused
unchanged across every chunk). draft_data's card reference for the two
metrics built here is a VALUE in the data — the 'pick' column names one
card per row — so there is no header-derived CardColumnSet list to
hand accumulate(); instead, DraftMetricScanner resolves each chunk's
'pick' column itself (via pick_name_cache.py's PickNameCache, which
owns the CardBinder access and the cross-chunk resolution cache) and
passes the already-resolved result in. DraftMetric implementations
never talk to CardBinder directly, same division of responsibility as
Metric — only the shape of what they're handed differs.
"""

from typing import Protocol
from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult


class DraftMetric(Protocol):
    """Strategy: accumulate one numeric metric across chunks of draft_data.

    Implementations must be safe to call accumulate() many times in a
    row (once per chunk) before finalize() is ever called. Single-
    consumer to DraftMetricScanner, which is agnostic to which concrete
    implementation it's driving.
    """

    name: str

    def accumulate(self, chunk: pd.DataFrame, resolved_picks: pd.Series) -> None:
        """Update this metric's running accumulator with one chunk of rows.

        Inputs:
            chunk: one chunk of the raw draft_data CSV, as read by
                pandas.read_csv(..., chunksize=...).
            resolved_picks: index-aligned with chunk, one resolved
                nocab_uuid per row (or None where that row's 'pick'
                value couldn't be resolved — see
                pick_name_cache.py's PickNameCache, which produces
                this before accumulate() is ever called; this method
                never queries CardBinder itself).
        Output: none.
        Side effects: mutates this metric's internal accumulator state.
        Exceptions: none expected from well-formed input.
        """
        ...

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn this metric's accumulated state into final MetricResult rows.

        Inputs: none (uses this metric's internal accumulator state).
        Output: one MetricResult per nocab_uuid this metric has seen
            enough data to produce a result for.
        Side effects: none.
        Exceptions: none expected.
        """
        ...

    def save_state(self) -> dict:
        """Snapshot this metric's accumulator state as serializable data.

        Memento (PATTERNS.md) — the returned dict must be JSON-
        serializable, which means any UUID-keyed internal state must
        be stringified here, not left as UUID objects.

        Inputs: none (uses this metric's internal accumulator state).
        Output: a JSON-serializable dict capturing enough state to
            fully restore this metric via load_state() later.
        Side effects: none.
        Exceptions: none expected.
        """
        ...

    def load_state(self, state: dict) -> None:
        """Restore this metric's accumulator state from a save_state() snapshot.

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
