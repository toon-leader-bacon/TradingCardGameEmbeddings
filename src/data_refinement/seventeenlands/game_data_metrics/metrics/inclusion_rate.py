"""Inclusion rate = fraction of all games in the file that had a given
card in the deck.

See src/data_refinement/seventeenlands/game_data_metrics/metric.py
for the shared Metric interface this implements. Deliberately does
not touch the "won" column at all — a structurally different metric
shape from the win-rate family (metrics/win_rate/): the denominator
here is every game processed, not a per-card filtered subset, so this
metric tracks one shared "games processed" counter plus a per-card
"times included" counter, not a per-card wins/games pair.

Note the human explicitly flagged this metric as likely low-value for
training (it probably just reduces to "how rare is this card in a
booster," i.e. printed rarity) but wanted it built anyway to
demonstrate a metric shape that doesn't depend on "won".
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "inclusion_rate"


class InclusionRateMetric:
    """Inclusion rate = (games with this card in the deck) / (all games).

    Single-consumer to whatever MetricScanner it's constructed for — a
    fresh instance is expected per MetricScanner.scan() call (or one
    being resumed via load_state() from a checkpoint of that same
    scan).
    """

    name: str = _METRIC_NAME

    def __init__(self, expansion: str, format_code: str) -> None:
        """
        Inputs:
            expansion: 17lands expansion code this metric's raw CSV
                belongs to — stamped onto every MetricResult this
                metric produces.
            format_code: 17lands format code — stamped onto every
                MetricResult this metric produces. Named format_code,
                not format, to avoid shadowing the format() builtin.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._expansion = expansion
        self._format_code = format_code
        self._games_processed = 0
        self._inclusions: dict[UUID, int] = {}

    def accumulate(
        self, chunk: pd.DataFrame, card_columns: list[CardColumnSet]
    ) -> None:
        """Update the shared games-processed count and every card's
        inclusion count, from one chunk.

        self._games_processed increases by len(chunk) exactly once per
        call — it is the same denominator for every card, unlike the
        win-rate family's per-card denominators. For each
        CardColumnSet in card_columns, if chunk[card_column_set.deck] > 0
        for at least one row, self._inclusions[uuid] is incremented by
        that count (a vectorized pandas boolean mask, summed) — a card
        with zero inclusions in this chunk is left untouched in
        self._inclusions, matching the win-rate family's "absence
        means never observed" convention (see finalize()).

        Inputs:
            chunk: one chunk of the raw game_data CSV — must include
                every CardColumnSet's deck column. Does not need a
                "won" column — this metric never reads it.
            card_columns: every resolved card's column names for this
                scan, unchanged across calls.
        Output: none.
        Side effects: increments self._games_processed unconditionally
            (once per call, regardless of any card's inclusion count);
            increments self._inclusions in place for cards with a
            nonzero inclusion count this chunk, adding to whatever was
            already accumulated from prior calls.
        Exceptions: none expected from well-formed input.
        """
        self._games_processed += len(chunk)

        for card_column_set in card_columns:
            included = chunk[card_column_set.deck] > 0
            inclusions_this_chunk = int(included.sum())
            if inclusions_this_chunk == 0:
                continue

            uuid = card_column_set.nocab_uuid
            self._inclusions[uuid] = (
                self._inclusions.get(uuid, 0) + inclusions_this_chunk
            )

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn the accumulated counts into MetricResult rows.

        value = self._inclusions[uuid] / self._games_processed;
        sample_size = self._games_processed (the same for every card
        in the result, since inclusion rate's denominator is global,
        not per-card — unlike the win-rate family, where sample_size
        varies per card). A card never present in self._inclusions
        (zero inclusions across the whole scan) produces no result at
        all — same "absence means never observed, not an explicit
        zero" convention as the win-rate family, and avoids a
        finalize()-time dependency on knowing the full resolved-card
        set separately from what accumulate() has actually seen.

        Inputs: none (uses self._games_processed, self._inclusions).
        Output: one MetricResult per nocab_uuid present in
            self._inclusions. Empty if self._games_processed is 0
            (accumulate() was never called).
        Side effects: none.
        Exceptions: none expected.
        """
        return {
            uuid: MetricResult(
                nocab_uuid=uuid,
                metric_name=self.name,
                value=count / self._games_processed,
                sample_size=self._games_processed,
                expansion=self._expansion,
                format=self._format_code,
            )
            for uuid, count in self._inclusions.items()
        }

    def save_state(self) -> dict:
        """Snapshot accumulated state as JSON-serializable data.

        UUID keys are stringified (JSON object keys must be strings).

        Inputs: none (uses self._games_processed, self._inclusions).
        Output: {"games_processed": <int>, "inclusions": {<uuid str>:
            <int>, ...}}.
        Side effects: none.
        Exceptions: none.
        """
        return {
            "games_processed": self._games_processed,
            "inclusions": {
                str(uuid): count for uuid, count in self._inclusions.items()
            },
        }

    def load_state(self, state: dict) -> None:
        """Restore accumulated state from a save_state() snapshot.

        Inputs:
            state: a dict previously returned by this class's own
                save_state().
        Output: none.
        Side effects: overwrites self._games_processed and
            self._inclusions.
        Exceptions: raises if state isn't shaped as save_state() would
            have produced.
        """
        self._games_processed = state["games_processed"]
        self._inclusions = {
            UUID(uuid_str): count for uuid_str, count in state["inclusions"].items()
        }
