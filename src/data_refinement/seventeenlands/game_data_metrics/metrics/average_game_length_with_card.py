"""Average game length (in turns), per card, across games it was in.

See src/data_refinement/seventeenlands/game_data_metrics/metric.py
for the shared Metric interface this implements. Demonstrates that
MetricResult.value is a generic float, not assumed to be a 0-1
probability — this is a plain average of the "num_turns" metadata
column, conditioned per-card on inclusion (deck_<name> > 0).

Note: see average_copies_when_included.py's module docstring for why
this metric (despite a very similar "running (sum, count) pair" shape)
is kept as its own independent class rather than sharing a base with
that one — "rule of two," not yet worth extracting.
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "average_game_length_with_card"


class AverageGameLengthWithCardMetric:
    """Average num_turns, per card, across games where it was included.

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
        self._turn_sum: dict[UUID, int] = {}
        self._game_count: dict[UUID, int] = {}

    def accumulate(
        self, chunk: pd.DataFrame, card_columns: list[CardColumnSet]
    ) -> None:
        """Sum turn counts and count games, per card, from one chunk.

        For each CardColumnSet in card_columns: rows where
        chunk[card_column_set.deck] > 0 are this card's games this
        chunk — this metric sums chunk["num_turns"] over exactly those
        rows into self._turn_sum, and counts how many such rows there
        were into self._game_count. A card with zero games in this
        chunk is left entirely untouched in both dicts — same
        "absence means never observed" convention as WinRateMetric,
        for the same reason.

        Inputs:
            chunk: one chunk of the raw game_data CSV — must include a
                "num_turns" column and every CardColumnSet's deck
                column.
            card_columns: every resolved card's column names for this
                scan, unchanged across calls.
        Output: none.
        Side effects: increments self._turn_sum and self._game_count
            in place for cards with at least one game this chunk,
            adding to whatever was already accumulated from prior
            calls.
        Exceptions: none expected from well-formed input.
        """
        for card_column_set in card_columns:
            included = chunk[card_column_set.deck] > 0
            game_count_this_chunk = int(included.sum())
            if game_count_this_chunk == 0:
                continue
            turn_sum_this_chunk = int(chunk.loc[included, "num_turns"].sum())

            uuid = card_column_set.nocab_uuid
            self._turn_sum[uuid] = self._turn_sum.get(uuid, 0) + turn_sum_this_chunk
            self._game_count[uuid] = (
                self._game_count.get(uuid, 0) + game_count_this_chunk
            )

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn accumulated turn sums/game counts into MetricResult rows.

        value = self._turn_sum[uuid] / self._game_count[uuid];
        sample_size = self._game_count[uuid].

        Inputs: none (uses self._turn_sum, self._game_count).
        Output: one MetricResult per nocab_uuid present in
            self._game_count.
        Side effects: none.
        Exceptions: none expected.
        """
        return {
            uuid: MetricResult(
                nocab_uuid=uuid,
                metric_name=self.name,
                value=self._turn_sum[uuid] / count,
                sample_size=count,
                expansion=self._expansion,
                format=self._format_code,
            )
            for uuid, count in self._game_count.items()
        }

    def save_state(self) -> dict:
        """Snapshot accumulated state as JSON-serializable data.

        UUID keys are stringified (JSON object keys must be strings).

        Inputs: none (uses self._turn_sum, self._game_count).
        Output: {"turn_sum": {<uuid str>: <int>, ...}, "game_count":
            {<uuid str>: <int>, ...}}.
        Side effects: none.
        Exceptions: none.
        """
        return {
            "turn_sum": {str(uuid): count for uuid, count in self._turn_sum.items()},
            "game_count": {
                str(uuid): count for uuid, count in self._game_count.items()
            },
        }

    def load_state(self, state: dict) -> None:
        """Restore accumulated state from a save_state() snapshot.

        Inputs:
            state: a dict previously returned by this class's own
                save_state().
        Output: none.
        Side effects: overwrites self._turn_sum and self._game_count.
        Exceptions: raises if state isn't shaped as save_state() would
            have produced.
        """
        self._turn_sum = {
            UUID(uuid_str): count for uuid_str, count in state["turn_sum"].items()
        }
        self._game_count = {
            UUID(uuid_str): count for uuid_str, count in state["game_count"].items()
        }
