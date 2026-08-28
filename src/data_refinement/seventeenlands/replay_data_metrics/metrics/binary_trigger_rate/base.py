"""Shared base for ReplayMetrics computing "fraction of games where a
card appeared in some resolved hand list that also had some outcome
true", differing only in which ResolvedHands field triggers inclusion
and how the outcome itself is determined.

Private module (leading underscore on the class, not the file — this
family now lives in its own metrics/binary_trigger_rate/ subdirectory,
so the directory itself signals "shared, family-internal" instead of a
leading-underscore filename convention) — shared only by this
directory's own opening_hand_win_rate.py /
candidate_hand_mulligan_rate.py, never imported outside
metrics/binary_trigger_rate/. Mirrors
game_data_metrics/metrics/win_rate/base.py's shape (a private,
non-Protocol base; public subclasses set `name` plus whatever varies)
but is NOT the same class — game_data_metrics' base reads a
CardColumnSet's per-column counts via vectorized pandas boolean masks
(identity is known from the header up front); this base reads a
per-row LIST of nocab_uuids off a arena_id_cache.ResolvedHands object
(card identity is only known per row, resolved by ArenaIdCache), so its
inner loop is over that row's small, fixed-size hand list, not a
vectorized mask over card_columns.

_outcome() is a method, not a fixed column-name string like
_hand_field, because "the outcome" doesn't have one consistent home:
OpeningHandWinRateMetric's outcome ("won") is a raw chunk column, but
CandidateHandMulliganRateMetric's outcome ("mulliganed") is a derived
ResolvedHands field, not a chunk column at all — there's no single
string that names both uniformly, so each subclass overrides a small
method instead.
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.metric_result import MetricResult


class _BinaryTriggerRateMetric:
    """Rate = fraction of games where a card appeared in
    getattr(<row's ResolvedHands>, _hand_field) that also had
    _outcome() true for that row, for whichever field/outcome a
    subclass selects.

    Not a complete ReplayMetric on its own: subclasses must set `name`
    (a class attribute, per ReplayMetric's Protocol) and `_hand_field`
    (which arena_id_cache.ResolvedHands field selects the trigger list
    for this row, e.g. "opening_hand"), and override `_outcome()` (see
    this module's docstring for why it's a method, not a string).
    """

    name: str  # set by each subclass
    _hand_field: str  # set by each subclass

    def __init__(self, expansion: str, format_code: str) -> None:
        """
        Inputs:
            expansion: 17lands expansion code this metric's raw CSV
                belongs to (e.g. "MSH") — stamped onto every
                MetricResult this metric produces.
            format_code: 17lands format code (e.g. "PremierDraft") —
                stamped onto every MetricResult this metric produces.
                Named format_code, not format, to avoid shadowing the
                format() builtin (MetricResult's own field is still
                named format).
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._expansion = expansion
        self._format_code = format_code
        self._positive: dict[UUID, int] = {}
        self._total: dict[UUID, int] = {}

    def _outcome(self, chunk: pd.DataFrame, resolved: pd.Series) -> pd.Series:
        """Subclass hook: this chunk's per-row boolean outcome.

        Overridden by each subclass — e.g. OpeningHandWinRateMetric
        returns chunk["won"]; CandidateHandMulliganRateMetric returns
        resolved.map(lambda r: r.mulliganed).

        Inputs:
            chunk: the raw chunk, as passed to accumulate().
            resolved: the ResolvedHands Series, as passed to
                accumulate() — index-aligned with chunk.
        Output: a boolean-ish Series, index-aligned with chunk/resolved.
        Side effects: none.
        Exceptions: none expected.
        """
        raise NotImplementedError

    def accumulate(self, chunk: pd.DataFrame, resolved: pd.Series) -> None:
        """Update per-card positive/total counts, from one chunk.

        For each row, for each nocab_uuid in
        getattr(resolved[row], self._hand_field) (deduplicated already
        by ArenaIdCache — see arena_id_cache.py): that card's total
        count is incremented by 1, and its positive count is
        additionally incremented by 1 if self._outcome(chunk, resolved)
        is true for that row. A row whose hand list is empty
        contributes nothing for any card.

        Inputs:
            chunk: one chunk of the raw replay_data CSV.
            resolved: a pd.Series of arena_id_cache.ResolvedHands,
                index-aligned with chunk.
        Output: none.
        Side effects: increments this metric's internal positive/total
            counts in place (adds this chunk's counts to whatever was
            already accumulated — never overwrites/resets either). A
            card with zero observed rows in this chunk is left entirely
            untouched — same "don't touch the denominator for zero
            observations" invariant as game_data_metrics'
            _BinaryTriggerWinRateMetric (see its accumulate() docstring
            for why this matters for finalize()).
        Exceptions: none expected from well-formed input.
        """
        hand_lists = resolved.map(lambda r: getattr(r, self._hand_field))
        outcomes = self._outcome(chunk, resolved)

        for hand, outcome in zip(hand_lists, outcomes):
            for uuid in hand:
                self._total[uuid] = self._total.get(uuid, 0) + 1
                if outcome:
                    self._positive[uuid] = self._positive.get(uuid, 0) + 1

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn accumulated positive/total counts into MetricResult rows.

        value = positive count / total count; sample_size = total
        count. No minimum-sample-size cutoff — every card with at
        least one observed row gets a result.

        Inputs: none (uses this metric's internal accumulator state).
        Output: one MetricResult per nocab_uuid with at least one
            observed row, with metric_name=self.name.
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

        Inputs: none (uses this metric's internal accumulator state).
        Output: a dict shaped consistently for every subclass, e.g.
            {"positive": {<uuid str>: <int>, ...}, "total": {<uuid
            str>: <int>, ...}}.
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
