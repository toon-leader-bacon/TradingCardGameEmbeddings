"""Shared base for Metrics computing "win rate when column X was
true", differing only in which CardColumnSet field triggers inclusion.

Private module (leading underscore on the class, not the file — this
family now lives in its own metrics/win_rate/ subdirectory, so the
directory itself signals "shared, family-internal" instead of a
leading-underscore filename convention) — shared only by this
directory's own win_rate.py / drawn_win_rate.py /
opening_hand_win_rate.py, never imported outside metrics/win_rate/.
Extracted after the third such metric (opening_hand_win_rate) was
added, not preemptively — this project's own stated convention (see
src/data_retrieval/README.md's rationale for download_to_file.py:
"extracted after the same block showed up independently in three
downloaders... the rule of three case, not a preemptive one"). Before
this third metric, WinRateMetric stood alone as its own full
implementation in win_rate.py; two near-identical metrics would not by
itself have justified this extraction.

_BinaryTriggerWinRateMetric is not itself a Metric (it deliberately
doesn't set `name` — see class docstring) — the three public
subclasses (WinRateMetric, DrawnWinRateMetric, OpeningHandWinRateMetric)
are the actual Metrics, each setting `name` and `_trigger_column` and
inheriting everything else unchanged.
"""

from typing import Callable
from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.metric_result import MetricResult


class _BinaryTriggerWinRateMetric:
    """Win rate = fraction of games where <trigger column> was true
    that were also won, for whichever CardColumnSet field a subclass
    selects via _trigger_column.

    Not a complete Metric on its own: subclasses must set `name`
    (a class attribute, per Metric's Protocol) and
    `_trigger_column` — a staticmethod selecting which of
    CardColumnSet's own fields ("deck", "drawn", or "opening_hand")
    to read for "was this card present in this game, by this
    measure." A typed callable rather than a bare attribute-name
    string (e.g. `_trigger_column_attr = "dekc"`, a typo that would
    only surface as an AttributeError at the first accumulate() call)
    — a callable selector lets a type checker verify the selected
    field actually exists on CardColumnSet.
    """

    name: str  # set by each subclass
    _trigger_column: Callable[[CardColumnSet], str]  # set by each subclass,
    # e.g. staticmethod(lambda columns: columns.deck)

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
        self._wins: dict[UUID, int] = {}
        self._games: dict[UUID, int] = {}

    def accumulate(
        self, chunk: pd.DataFrame, card_columns: list[CardColumnSet]
    ) -> None:
        """Update wins/games counts for every card, from one chunk.

        For each CardColumnSet in card_columns: a game counts toward
        that card's denominator if
        chunk[self._trigger_column(card_column_set)] > 0; it
        additionally counts toward the numerator if chunk["won"] is
        also True for that row. Both counts are computed via
        vectorized pandas boolean masks summed over the whole chunk.

        Inputs:
            chunk: one chunk of the raw game_data CSV — must include a
                "won" column and every CardColumnSet's trigger column.
            card_columns: every resolved card's column names for this
                scan, unchanged across calls.
        Output: none.
        Side effects: increments self._wins and self._games in place
            (adds this chunk's counts to whatever was already
            accumulated — never overwrites/resets either dict). A card
            with zero observed games (by the trigger column) in this
            chunk is left entirely untouched — see WinRateMetric's
            prior implementation/tests for why unconditionally
            touching self._games would corrupt finalize()'s "every
            card in self._games has at least one observed game"
            invariant and cause a ZeroDivisionError there.
        Exceptions: none expected from well-formed input.
        """
        for card_column_set in card_columns:
            trigger_column = self._trigger_column(card_column_set)
            triggered = chunk[trigger_column] > 0
            games_this_chunk = int(triggered.sum())
            if games_this_chunk == 0:
                continue
            wins_this_chunk = int((triggered & chunk["won"]).sum())

            uuid = card_column_set.nocab_uuid
            self._games[uuid] = self._games.get(uuid, 0) + games_this_chunk
            self._wins[uuid] = self._wins.get(uuid, 0) + wins_this_chunk

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn accumulated wins/games counts into MetricResult rows.

        value = self._wins[uuid] / self._games[uuid]; sample_size =
        self._games[uuid]. No minimum-sample-size cutoff — every card
        with at least one observed game (by the trigger column) gets a
        result.

        Inputs: none (uses self._wins, self._games).
        Output: one MetricResult per nocab_uuid present in
            self._games, with metric_name=self.name.
        Side effects: none.
        Exceptions: none expected.
        """
        return {
            uuid: MetricResult(
                nocab_uuid=uuid,
                metric_name=self.name,
                value=self._wins[uuid] / games,
                sample_size=games,
                expansion=self._expansion,
                format=self._format_code,
            )
            for uuid, games in self._games.items()
        }

    def save_state(self) -> dict:
        """Snapshot wins/games as JSON-serializable data.

        UUID keys are stringified (JSON object keys must be strings).

        Inputs: none (uses self._wins, self._games).
        Output: {"wins": {<uuid str>: <int>, ...}, "games": {<uuid
            str>: <int>, ...}}.
        Side effects: none.
        Exceptions: none.
        """
        return {
            "wins": {str(uuid): count for uuid, count in self._wins.items()},
            "games": {str(uuid): count for uuid, count in self._games.items()},
        }

    def load_state(self, state: dict) -> None:
        """Restore wins/games from a save_state() snapshot.

        Inputs:
            state: a dict previously returned by this class's own
                save_state().
        Output: none.
        Side effects: overwrites self._wins and self._games.
        Exceptions: raises if state isn't shaped as save_state() would
            have produced.
        """
        self._wins = {
            UUID(uuid_str): count for uuid_str, count in state["wins"].items()
        }
        self._games = {
            UUID(uuid_str): count for uuid_str, count in state["games"].items()
        }
