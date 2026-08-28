"""Average pick_number, per card, across every pick of that card.

See src/data_refinement/seventeenlands/draft_data_metrics/draft_metric.py
for the shared DraftMetric interface this implements.

value = mean of the raw 'pick_number' column (0-indexed, as it appears
in the source data — not transformed or 1-indexed here) across every
row where that card was the one picked (resolved_picks == this card's
nocab_uuid).

Deliberately does NOT also track or combine pack_number. This was a
real design choice, not an oversight: pick_number (position within
whichever pack a card was taken from, e.g. "1st pick") is a meaningful
power-level signal regardless of which of the draft's packs it came
from — a 1st pick in pack 3 is still a strong signal, much like a 1st
pick in pack 1 — but WHICH pack a card was picked in (pack_number)
doesn't add much signal on its own. So no pack_number*N+pick_number
combined formula, and no separate pack_number tracking; plain average
pick_number is the metric judged worth building here.

Kept as its own independent class, not sharing a base with
pick_sideboard_rate.py despite a similar "average some existing row
column, grouped by resolved pick uuid" shape — matches this project's
established "rule of three, not preemptive" convention (see
game_data_metrics/metrics/average_copies_when_included.py's identical
reasoning for its own non-extraction) — this is only "rule of two."
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "average_pick_number"


class AveragePickNumberMetric:
    """Average pick_number, per card, across every pick of that card.

    Single-consumer to whatever DraftMetricScanner it's constructed
    for — a fresh instance is expected per DraftMetricScanner.scan()
    call (or one being resumed via load_state() from a checkpoint of
    that same scan).
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
        self._pick_number_sum: dict[UUID, int] = {}
        self._pick_count: dict[UUID, int] = {}

    def accumulate(self, chunk: pd.DataFrame, resolved_picks: pd.Series) -> None:
        """Sum pick_number and count picks, per card, from one chunk.

        For each distinct nocab_uuid present in resolved_picks this
        chunk (excluding unresolved rows, i.e. None/NaN entries): sums
        chunk["pick_number"] over exactly the rows where
        resolved_picks equals that uuid, into self._pick_number_sum,
        and counts how many such rows there were into
        self._pick_count. A card with zero picks in this chunk is left
        entirely untouched in both dicts — same "absence means never
        observed" convention as every game_data_metrics metric (see
        WinRateMetric's accumulate() docstring for why this matters:
        it keeps finalize()'s "every uuid in self._pick_count has at
        least one pick" invariant true, avoiding a ZeroDivisionError
        there).

        Inputs:
            chunk: one chunk of the raw draft_data CSV — must include
                a "pick_number" column.
            resolved_picks: index-aligned with chunk, one resolved
                nocab_uuid per row (or None where unresolved) — see
                pick_name_cache.py's PickNameCache, which produces
                this before accumulate() is ever called.
        Output: none.
        Side effects: increments self._pick_number_sum and
            self._pick_count in place for cards with at least one pick
            this chunk, adding to whatever was already accumulated
            from prior calls.
        Exceptions: none expected from well-formed input.
        """
        for uuid in resolved_picks.dropna().unique():
            mask = resolved_picks == uuid
            pick_count_this_chunk = int(mask.sum())
            pick_number_sum_this_chunk = int(chunk.loc[mask, "pick_number"].sum())

            self._pick_number_sum[uuid] = (
                self._pick_number_sum.get(uuid, 0) + pick_number_sum_this_chunk
            )
            self._pick_count[uuid] = (
                self._pick_count.get(uuid, 0) + pick_count_this_chunk
            )

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn accumulated pick_number sums/counts into MetricResult rows.

        value = self._pick_number_sum[uuid] / self._pick_count[uuid];
        sample_size = self._pick_count[uuid].

        Inputs: none (uses self._pick_number_sum, self._pick_count).
        Output: one MetricResult per nocab_uuid present in
            self._pick_count.
        Side effects: none.
        Exceptions: none expected.
        """
        return {
            uuid: MetricResult(
                nocab_uuid=uuid,
                metric_name=self.name,
                value=self._pick_number_sum[uuid] / count,
                sample_size=count,
                expansion=self._expansion,
                format=self._format_code,
            )
            for uuid, count in self._pick_count.items()
        }

    def save_state(self) -> dict:
        """Snapshot accumulated state as JSON-serializable data.

        UUID keys are stringified (JSON object keys must be strings).

        Inputs: none (uses self._pick_number_sum, self._pick_count).
        Output: {"pick_number_sum": {<uuid str>: <int>, ...},
            "pick_count": {<uuid str>: <int>, ...}}.
        Side effects: none.
        Exceptions: none.
        """
        return {
            "pick_number_sum": {
                str(uuid): count for uuid, count in self._pick_number_sum.items()
            },
            "pick_count": {
                str(uuid): count for uuid, count in self._pick_count.items()
            },
        }

    def load_state(self, state: dict) -> None:
        """Restore accumulated state from a save_state() snapshot.

        Inputs:
            state: a dict previously returned by this class's own
                save_state().
        Output: none.
        Side effects: overwrites self._pick_number_sum and
            self._pick_count.
        Exceptions: raises if state isn't shaped as save_state() would
            have produced.
        """
        self._pick_number_sum = {
            UUID(uuid_str): count
            for uuid_str, count in state["pick_number_sum"].items()
        }
        self._pick_count = {
            UUID(uuid_str): count for uuid_str, count in state["pick_count"].items()
        }
