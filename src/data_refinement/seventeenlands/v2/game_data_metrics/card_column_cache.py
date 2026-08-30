"""Shared lazy card-column lookup for v2 game_data-sourced metrics.

Extracted directly at authoring time (not after a third instance
accreted organically) because DeckOutcomeMetric and WinRateMetric were
both written in the same design pass, already known to need the exact
same lazy-resolve-from-a-chunk's-own-header-then-cache logic — this is
PRINCIPLES.md section 2's "logic that's actually the same" case, not
the "superficially similar, prefer duplication" case the project's
"rule of three, not preemptive" convention (see e.g. win_rate/base.py)
otherwise defaults to for logic that only turns out to be identical
after a third, independently-written instance shows up later.
"""

import warnings

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.game_data_metrics.column_lookup import (
    find_card_columns,
)
from src.schema.game_id import GameId


class CardColumnCache:
    """Lazily looks up a scan's card columns from the first chunk it
    sees, caching the result for every later call.

    Single-consumer-per-scan — one instance is held privately by one
    Metric instance for that metric's whole lifetime, not shared
    across metrics (each metric's own accumulate() calls its own
    cache's card_columns() with whatever chunk it's currently
    processing).
    """

    def __init__(self, card_binder: CardBinder, source_game: GameId) -> None:
        """
        Inputs:
            card_binder: registry to look card names up against.
            source_game: which game card_binder's aliases should be
                looked up against.
        Output: none (constructor).
        Side effects: none — no lookup happens until card_columns() is
            first called.
        Exceptions: none.
        """
        self._card_binder = card_binder
        self._source_game = source_game
        self._card_columns: list[CardColumnSet] | None = None
        self._unresolved_column_names: list[str] = []

    def card_columns(self, chunk: pd.DataFrame) -> list[CardColumnSet]:
        """Look up chunk's header against card_binder, once, caching the result.

        Every deck_<name>/opening_hand_<name>/etc. column set is the
        same across every chunk of one scan, so looking this up once
        (from whichever chunk happens to be first) and caching it is
        correct and sufficient — later calls, even with a different
        chunk argument, return the cached result unchanged without
        re-deriving anything.

        Inputs:
            chunk: the chunk currently being processed by the calling
                metric — only its .columns are read (converted to
                list[str] via chunk.columns.tolist(), since
                column_lookup.find_card_columns() expects a plain
                list, not a pandas Index), and only on the first call.
        Output: every card column set this scan's header contains that
            card_binder could resolve.
        Side effects: on the first call, populates this cache's
            internal state (including unresolved_column_names, below)
            and warns (does not raise) if any column names couldn't be
            resolved; a no-op on every later call.
        Exceptions: none expected from well-formed input.
        """
        if self._card_columns is not None:
            return self._card_columns

        header = chunk.columns.tolist()
        resolution = find_card_columns(header, self._card_binder, self._source_game)
        self._card_columns = resolution.resolved
        self._unresolved_column_names = resolution.unresolved_names
        if self._unresolved_column_names:
            warnings.warn(
                f"Could not resolve {len(self._unresolved_column_names)} card name(s) "
                f"against CardBinder: {self._unresolved_column_names}",
                RuntimeWarning,
            )
        return self._card_columns

    @property
    def unresolved_column_names(self) -> list[str]:
        """Card names this cache's header lookup couldn't resolve.

        Empty until card_columns() has been called at least once.

        Inputs: none.
        Output: every unresolved card name from this scan's header, in
            the order column_lookup.find_card_columns() reported them.
        Side effects: none.
        Exceptions: none.
        """
        return self._unresolved_column_names
