"""Average number of copies of a card run, in decks that ran it at all.

See src/data_refinement/seventeenlands/game_data_metrics/metric.py
for the shared Metric interface this implements. Demonstrates that
accumulate() can sum an actual numeric column value, not just count a
boolean mask — a card you can run multiple copies of has
deck_<name> > 1 in some games, and this metric's value is the mean of
that count across games where the card was included at all (not
averaged across every game in the file — see finalize()).

Note: this metric and average_game_length_with_card.py share a similar
"running (sum, count) pair conditioned on deck_<name> > 0" shape, but
summing a different source column each (this metric: the card's own
deck column; that one: the "num_turns" metadata column). Left as two
independent classes rather than extracted into a shared base — unlike
the three win-rate metrics (metrics/win_rate/), which hit this
project's stated three-occurrence threshold for extraction, these two
alone are only "rule of two." Revisit if a third "average of column Y
conditioned on deck_<name> > 0" metric is added later.
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "average_copies_when_included"


class AverageCopiesWhenIncludedMetric:
    """Average copies run, per card, across games where it was included.

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
        self._copy_sum: dict[UUID, int] = {}
        self._inclusion_count: dict[UUID, int] = {}

    def accumulate(
        self, chunk: pd.DataFrame, card_columns: list[CardColumnSet]
    ) -> None:
        """Sum copies-run and count inclusions, per card, from one chunk.

        For each CardColumnSet in card_columns: rows where
        chunk[card_column_set.deck] > 0 are "inclusions" — this
        metric sums chunk[card_column_set.deck] over exactly those rows
        (the actual copy count, not just a boolean — e.g. 2 copies run
        in one game contributes 2, not 1) into self._copy_sum, and
        counts how many such rows there were into
        self._inclusion_count. A card with zero inclusions in this
        chunk is left entirely untouched in both dicts — same
        "absence means never observed" convention as WinRateMetric,
        for the same reason (avoids a spurious zero-count entry that
        would corrupt finalize()'s division).

        Inputs:
            chunk: one chunk of the raw game_data CSV — must include
                every CardColumnSet's deck column.
            card_columns: every resolved card's column names for this
                scan, unchanged across calls.
        Output: none.
        Side effects: increments self._copy_sum and
            self._inclusion_count in place for cards with at least one
            inclusion this chunk, adding to whatever was already
            accumulated from prior calls.
        Exceptions: none expected from well-formed input.
        """
        for card_column_set in card_columns:
            included = chunk[card_column_set.deck] > 0
            inclusion_count_this_chunk = int(included.sum())
            if inclusion_count_this_chunk == 0:
                continue
            copy_sum_this_chunk = int(chunk.loc[included, card_column_set.deck].sum())

            uuid = card_column_set.nocab_uuid
            self._copy_sum[uuid] = self._copy_sum.get(uuid, 0) + copy_sum_this_chunk
            self._inclusion_count[uuid] = (
                self._inclusion_count.get(uuid, 0) + inclusion_count_this_chunk
            )

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn accumulated copy sums/inclusion counts into MetricResult rows.

        value = self._copy_sum[uuid] / self._inclusion_count[uuid];
        sample_size = self._inclusion_count[uuid].

        Inputs: none (uses self._copy_sum, self._inclusion_count).
        Output: one MetricResult per nocab_uuid present in
            self._inclusion_count.
        Side effects: none.
        Exceptions: none expected.
        """
        return {
            uuid: MetricResult(
                nocab_uuid=uuid,
                metric_name=self.name,
                value=self._copy_sum[uuid] / count,
                sample_size=count,
                expansion=self._expansion,
                format=self._format_code,
            )
            for uuid, count in self._inclusion_count.items()
        }

    def save_state(self) -> dict:
        """Snapshot accumulated state as JSON-serializable data.

        UUID keys are stringified (JSON object keys must be strings).

        Inputs: none (uses self._copy_sum, self._inclusion_count).
        Output: {"copy_sum": {<uuid str>: <int>, ...},
            "inclusion_count": {<uuid str>: <int>, ...}}.
        Side effects: none.
        Exceptions: none.
        """
        return {
            "copy_sum": {str(uuid): count for uuid, count in self._copy_sum.items()},
            "inclusion_count": {
                str(uuid): count for uuid, count in self._inclusion_count.items()
            },
        }

    def load_state(self, state: dict) -> None:
        """Restore accumulated state from a save_state() snapshot.

        Inputs:
            state: a dict previously returned by this class's own
                save_state().
        Output: none.
        Side effects: overwrites self._copy_sum and
            self._inclusion_count.
        Exceptions: raises if state isn't shaped as save_state() would
            have produced.
        """
        self._copy_sum = {
            UUID(uuid_str): count for uuid_str, count in state["copy_sum"].items()
        }
        self._inclusion_count = {
            UUID(uuid_str): count
            for uuid_str, count in state["inclusion_count"].items()
        }
