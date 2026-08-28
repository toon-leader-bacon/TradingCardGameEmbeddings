"""Opponent response rate = of every OWN-TURN cast instance of a card
(main-phase-style casts; see below for why off-turn casts are
excluded), the fraction where the OPPOSING side also had an off-turn
(flash-speed) cast recorded during that same turn — a turn-granularity
proxy for "does this bait/demand an answer," not a claim of direct
causal response (multiple things can happen in one turn; this is
expected to be noisy and wash out at volume, per this container's
design discussion).

Scoped to CastEvent.own_turn=True instances only (a real scope limit,
not an approximation): for an off-turn/flash cast, CastEvent.turn is
the OTHER side's turn count, not the caster's own (see
cast_event_scanner.py's CastEvent.own_turn docstring) — there is no
well-defined "same turn" column to check for the response signal in
that case, so off-turn casts contribute no observations to this metric
at all, rather than being checked against a column that doesn't
actually correspond to when they happened. Caught during
design-recipe-implement's full-pass review (file-critic) — an earlier
draft ignored own_turn and computed a nonsensical column lookup for
roughly half of all cast instances.

Deliberately NOT built on metrics/binary_trigger_rate/base.py's
_BinaryTriggerRateMetric, and NOT built on
metrics/cast_event_average/base.py's _CastEventAverageMetric, despite
being conceptually a rate over cast instances: _BinaryTriggerRateMetric
assumes at most one trigger-list per ROW (a resolved hand); this
metric's trigger is per CAST INSTANCE (a card cast 3 times in one row
is 3 independent trigger events, same as
metrics/cast_event_average/'s metrics). And unlike either shared
family, this metric's "outcome" column name is only known per-instance
(it depends on which side cast and which turn — see accumulate()'s
docstring) rather than being a single fixed column/field known at
class-definition time. Forcing this into either existing base would
mean generalizing both to cover a shape neither currently needs —
accepted as its own small standalone implementation instead
(PRINCIPLES.md section 2: prefer duplication over a coupled abstraction
that doesn't cleanly fit).
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.metric_result import MetricResult
from src.data_refinement.seventeenlands.replay_data_metrics.cast_event_scanner import (
    CastEventScanner,
    off_turn_response_column,
)
from src.schema.game_id import GameId

_METRIC_NAME = "opponent_response_rate"


class OpponentResponseRateMetric:
    """Response rate, pooled across both sides and every cast instance.

    Constructor takes card_binder/source_game in addition to
    expansion/format_code — same deviation, for the same reason, as
    _CastEventAverageMetric's constructor (see that class's docstring):
    this metric resolves its own turn-indexed columns via its own
    CastEventScanner, rather than receiving already-resolved data from
    ReplayMetricScanner.
    """

    name: str = _METRIC_NAME
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
        Side effects: constructs this metric's own CastEventScanner —
            no CardBinder queries happen until the first accumulate()
            call.
        Exceptions: none.
        """
        self._expansion = expansion
        self._format_code = format_code
        self._cast_event_scanner = CastEventScanner(card_binder, source_game)
        self._positive: dict[UUID, int] = {}
        self._total: dict[UUID, int] = {}

    def accumulate(self, chunk: pd.DataFrame, resolved: pd.Series) -> None:
        """Update per-card positive/total counts, from one chunk.

        For each row, finds every CastEvent via
        self._cast_event_scanner.find_cast_events(chunk), and considers
        ONLY events with own_turn=True — see CastEvent.own_turn's
        docstring: for an off-turn/flash-speed cast, event.turn is the
        OTHER side's turn count, not side's own, so
        f"{side}_turn_{turn}_..." would name a column that has no
        defined relationship to the turn the cast actually happened on.
        Restricting to own_turn=True events keeps this metric's
        "response during the same turn" check meaningful, at the cost
        of narrowing its scope to main-phase-style casts — off-turn
        casts by a card don't contribute any observations to this
        metric at all (a real scope limit, not an approximation; see
        replay_data_metrics/README.md, which already flags this
        metric's "response" definition as not fully validated). For
        each qualifying event (side S, turn T, card):
        that card's total count is incremented by 1; its positive
        count is additionally incremented by 1 if chunk's
        off_turn_response_column(S, T) column (see cast_event_scanner.py
        — the one place that owns this naming scheme; this metric asks
        it for the column name rather than reconstructing the format
        string itself) is non-empty for that row (i.e. the opposing
        side had an off-turn cast recorded during S's own turn T — see
        cast_event_scanner.py's module docstring for why this column
        naming pattern is the confirmed "response" signal).

        Inputs:
            chunk: one chunk of the raw replay_data CSV — must include
                every column cast_event_scanner.py's
                CastEventScanner reads, plus every
                <side>_turn_<N>_<other_side>_instants_sorceries_cast
                column this method reads directly.
            resolved: unused by this metric (its resolved data comes
                from its own CastEventScanner, not ArenaIdCache's
                ResolvedHands) — accepted anyway to satisfy
                ReplayMetric's Protocol signature.
        Output: none.
        Side effects: increments this metric's internal positive/total
            counts in place; mutates this metric's CastEventScanner
            cache/unresolved_ids as a side effect of resolving this
            chunk's cast columns. A card with zero cast instances in
            this chunk is left entirely untouched (same "don't touch
            the denominator for zero observations" invariant as
            _BinaryTriggerRateMetric.accumulate()).
        Exceptions: none expected from well-formed input.
        """
        events_per_row = self._cast_event_scanner.find_cast_events(chunk)

        for row_index, events in events_per_row.items():
            own_turn_events = [event for event in events if event.own_turn]
            if not own_turn_events:
                continue
            for event in own_turn_events:
                column = off_turn_response_column(event.side, event.turn)
                cell = chunk.at[row_index, column]
                responded: bool = pd.notna(cell) and cell != ""

                self._total[event.nocab_uuid] = self._total.get(event.nocab_uuid, 0) + 1
                if responded:
                    self._positive[event.nocab_uuid] = (
                        self._positive.get(event.nocab_uuid, 0) + 1
                    )

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn accumulated positive/total counts into MetricResult rows.

        value = positive count / total count; sample_size = total
        count. No minimum-sample-size cutoff — every card with at
        least one cast instance gets a result.

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
                value=self._positive.get(uuid, 0) / total,
                sample_size=total,
                expansion=self._expansion,
                format=self._format_code,
            )
            for uuid, total in self._total.items()
        }

    def save_state(self) -> dict:
        """Snapshot positive/total counts as JSON-serializable data.

        UUID keys are stringified (JSON object keys must be strings).
        Does NOT snapshot this metric's CastEventScanner cache — same
        rationale as _CastEventAverageMetric.save_state().

        Inputs: none (uses this metric's internal accumulator state).
        Output: a dict, e.g. {"positive": {<uuid str>: <int>, ...},
            "total": {<uuid str>: <int>, ...}}.
        Side effects: none.
        Exceptions: none.
        """
        return {
            "positive": {str(uuid): count for uuid, count in self._positive.items()},
            "total": {str(uuid): count for uuid, count in self._total.items()},
        }

    def load_state(self, state: dict) -> None:
        """Restore positive/total counts from a save_state() snapshot.

        Inputs:
            state: a dict previously returned by this class's own
                save_state().
        Output: none.
        Side effects: overwrites this metric's internal accumulator
            state.
        Exceptions: raises if state isn't shaped as save_state() would
            have produced.
        """
        self._positive = {
            UUID(uuid_str): count for uuid_str, count in state["positive"].items()
        }
        self._total = {
            UUID(uuid_str): count for uuid_str, count in state["total"].items()
        }
