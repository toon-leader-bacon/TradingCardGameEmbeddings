"""Shared base for ReplayMetrics computing "average <some per-cast-instance
value> across every cast instance of a card", differing only in which
value gets emitted per instance.

Private module (leading underscore on the class, not the file — this
family now lives in its own metrics/cast_event_average/ subdirectory)
— shared only by this directory's own average_turn_cast.py /
average_turns_remaining_post_cast.py, never imported outside
metrics/cast_event_average/. Mirrors
metrics/binary_trigger_rate/base.py's shape (a private, non-Protocol
base; public subclasses set `name` plus a small override) but is NOT
the same accumulator shape — that family accumulates positive/total
counts for a rate; this one accumulates sum/count for an average, over
CastEvents (one card can contribute MANY observations per row — every
cast instance is its own observation, per this session's settled
policy — not at most one per row the way a resolved hand list is).

Introduced directly with two subclasses (not extracted after a third),
same reasoning binary_trigger_rate/base.py's docstring already gives
for itself: both were designed together as one shape from the start.
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.metric_result import MetricResult
from src.data_refinement.seventeenlands.replay_data_metrics.cast_event_scanner import (
    CastEvent,
    CastEventScanner,
)
from src.schema.game_id import GameId


class _CastEventAverageMetric:
    """Average = mean of _value(event, num_turns) across every CastEvent
    found for a card, for whichever value a subclass computes via
    _value().

    Not a complete ReplayMetric on its own: subclasses must set `name`
    (a class attribute, per ReplayMetric's Protocol) and override
    `_value()` (see that method's docstring).

    Constructor takes card_binder/source_game in addition to
    expansion/format_code — a deviation from
    _BinaryTriggerRateMetric's constructor (which only takes
    expansion/format_code): those metrics receive already-resolved data
    from ReplayMetricScanner via ArenaIdCache; this metric resolves its
    own turn-indexed columns itself (see cast_event_scanner.py's
    module docstring for why that isn't centralized), so it needs a
    CardBinder reference of its own. Caller-level wiring (constructing
    this metric with the same card_binder/source_game a
    ReplayMetricScanner elsewhere uses) is the caller's concern, same
    convention as expansion/format_code.
    """

    name: str  # set by each subclass
    _cast_event_scanner: CastEventScanner  # constructed in __init__

    def __init__(
        self,
        expansion: str,
        format_code: str,
        card_binder: CardBinder,
        source_game: GameId,
    ) -> None:
        """
        Inputs:
            expansion: 17lands expansion code this metric's raw CSV
                belongs to (e.g. "MSH") — stamped onto every
                MetricResult this metric produces.
            format_code: 17lands format code (e.g. "PremierDraft") —
                stamped onto every MetricResult this metric produces.
            card_binder: registry this metric's own CastEventScanner
                resolves cast-column Arena IDs against.
            source_game: which game's alias namespace to resolve
                against.
        Output: none (constructor).
        Side effects: constructs this metric's own CastEventScanner
            (see cast_event_scanner.py) — no CardBinder queries
            happen until the first accumulate() call.
        Exceptions: none.
        """
        self._expansion = expansion
        self._format_code = format_code
        self._cast_event_scanner = CastEventScanner(card_binder, source_game)
        self._sum: dict[UUID, float] = {}
        self._count: dict[UUID, int] = {}

    def _value(self, event: CastEvent, num_turns: int) -> float:
        """Subclass hook: the value to emit for one cast instance.

        Overridden by each subclass — e.g. AverageTurnCastMetric
        returns event.turn; AverageTurnsRemainingPostCastMetric returns
        num_turns - event.turn.

        Inputs:
            event: one CastEvent found in some row.
            num_turns: that row's num_turns column value.
        Output: the numeric value this instance contributes to its
            card's running average.
        Side effects: none.
        Exceptions: none expected.
        """
        raise NotImplementedError

    def accumulate(self, chunk: pd.DataFrame, resolved: pd.Series) -> None:
        """Update per-card sum/count, from one chunk.

        For each row, finds every CastEvent via
        self._cast_event_scanner.find_cast_events(chunk) (called once
        per chunk, not per row), then for each event: adds
        self._value(event, row's num_turns) to that card's running sum
        and increments its running count by 1. A card with zero cast
        instances in this chunk is left entirely untouched (same
        "don't touch the denominator for zero observations" invariant
        as _BinaryTriggerRateMetric.accumulate()).

        Inputs:
            chunk: one chunk of the raw replay_data CSV — must include
                every column cast_event_scanner.py's
                CastEventScanner reads, plus "num_turns".
            resolved: unused by this metric (its resolved data comes
                from its own CastEventScanner, not ArenaIdCache's
                ResolvedHands) — accepted anyway to satisfy
                ReplayMetric's Protocol signature.
        Output: none.
        Side effects: increments this metric's internal sum/count in
            place (adds this chunk's observations to whatever was
            already accumulated); mutates this metric's CastEventScanner
            cache/unresolved_ids as a side effect of resolving this
            chunk's cast columns.
        Exceptions: none expected from well-formed input.
        """
        events_per_row = self._cast_event_scanner.find_cast_events(chunk)
        num_turns_per_row = chunk["num_turns"]

        for events, num_turns in zip(events_per_row, num_turns_per_row):
            for event in events:
                value = self._value(event, num_turns)
                self._sum[event.nocab_uuid] = (
                    self._sum.get(event.nocab_uuid, 0.0) + value
                )
                self._count[event.nocab_uuid] = self._count.get(event.nocab_uuid, 0) + 1

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn accumulated sum/count into MetricResult rows.

        value = sum / count; sample_size = count. No minimum-sample-
        size cutoff — every card with at least one cast instance gets
        a result.

        Inputs: none (uses this metric's internal accumulator state).
        Output: one MetricResult per nocab_uuid with at least one cast
            instance, with metric_name=self.name.
        Side effects: none.
        Exceptions: none expected.
        """
        return {
            uuid: MetricResult(
                nocab_uuid=uuid,
                metric_name=self.name,
                value=self._sum[uuid] / count,
                sample_size=count,
                expansion=self._expansion,
                format=self._format_code,
            )
            for uuid, count in self._count.items()
        }

    def save_state(self) -> dict:
        """Snapshot sum/count as JSON-serializable data.

        UUID keys are stringified (JSON object keys must be strings).

        Inputs: none (uses this metric's internal accumulator state).
        Output: a dict shaped consistently for every subclass, e.g.
            {"sum": {<uuid str>: <float>, ...}, "count": {<uuid str>:
            <int>, ...}}. Does NOT snapshot this metric's
            CastEventScanner cache — a resumed scan's scanner starts
            fresh (same rationale as ArenaIdCache's own checkpoint
            behavior — the cache is a perf optimization, not
            correctness-load-bearing state).
        Side effects: none.
        Exceptions: none.
        """
        return {
            "sum": {str(uuid): total for uuid, total in self._sum.items()},
            "count": {str(uuid): count for uuid, count in self._count.items()},
        }

    def load_state(self, state: dict) -> None:
        """Restore sum/count from a save_state() snapshot.

        Inputs:
            state: a dict previously returned by this class's own
                save_state().
        Output: none.
        Side effects: overwrites this metric's internal accumulator
            state.
        Exceptions: raises if state isn't shaped as save_state() would
            have produced.
        """
        self._sum = {UUID(uuid_str): total for uuid_str, total in state["sum"].items()}
        self._count = {
            UUID(uuid_str): count for uuid_str, count in state["count"].items()
        }
