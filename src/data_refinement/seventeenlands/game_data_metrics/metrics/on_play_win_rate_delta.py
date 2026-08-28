"""On-play win rate minus on-draw win rate, per card in deck.

See src/data_refinement/seventeenlands/game_data_metrics/metric.py
for the shared Metric interface this implements. Demonstrates a
genuinely derived value — a difference of two independently-computed
rates, not a direct ratio — conditioned on an additional metadata
column ("on_play") alongside "won" and deck_<name> > 0.

Two design decisions made here, flagged for human confirmation rather
than presented as the only obviously correct choices:

1. **sample_size for a delta metric.** Chosen: the SMALLER of the two
   conditioned sample sizes (games_on_play, games_on_draw), not their
   sum — the delta's reliability is bounded by whichever bucket has
   fewer observations (a card seen 500 times on the play but only 3
   times on the draw has a highly unreliable delta, which reporting
   sample_size=503 would obscure; sample_size=3 is honest about that).
   A downstream consumer applying its own noise cutoff (per this
   project's established "best-effort, honest sample_size" convention
   — see WinRateMetric's own docstring) sees the true limiting factor.
2. **Cards missing from either bucket entirely.** A card with zero
   observed games in EITHER the on-play or on-draw bucket has an
   undefined delta (one of the two rates is a division by zero) —
   such a card is omitted from finalize()'s results entirely, the
   same "absence means never observed / no defined value" convention
   every other metric in this container already uses for its own
   zero-sample-size case, rather than inventing a new degenerate-value
   convention just for this metric.
"""

from uuid import UUID

import pandas as pd

from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.metric_result import MetricResult

_METRIC_NAME = "on_play_win_rate_delta"


class OnPlayWinRateDeltaMetric:
    """(win rate on the play) - (win rate on the draw), per card in deck.

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
        self._wins_on_play: dict[UUID, int] = {}
        self._games_on_play: dict[UUID, int] = {}
        self._wins_on_draw: dict[UUID, int] = {}
        self._games_on_draw: dict[UUID, int] = {}

    def accumulate(
        self, chunk: pd.DataFrame, card_columns: list[CardColumnSet]
    ) -> None:
        """Update on-play and on-draw wins/games counts, per card, from
        one chunk.

        For each CardColumnSet in card_columns: in_deck =
        chunk[card_column_set.deck] > 0. Rows where in_deck &
        chunk["on_play"] update self._games_on_play /
        self._wins_on_play (the latter additionally masked by
        chunk["won"]); rows where in_deck & ~chunk["on_play"] update
        self._games_on_draw / self._wins_on_draw the same way. A row
        where the card isn't in the deck at all (in_deck is False)
        contributes to NEITHER bucket — both masks are conjoined with
        in_deck, so such a row is excluded from both by construction,
        same as every other metric in this container excluding
        not-in-deck rows entirely. All four dicts independently follow
        the same "leave untouched if this chunk contributed zero to
        this particular bucket" convention as WinRateMetric's
        self._games (see that class's accumulate() docstring for why)
        — a card can be untouched in self._games_on_draw while still
        being updated in self._games_on_play within the same chunk, or
        vice versa.

        Inputs:
            chunk: one chunk of the raw game_data CSV — must include
                "won", "on_play", and every CardColumnSet's deck
                column.
            card_columns: every resolved card's column names for this
                scan, unchanged across calls.
        Output: none.
        Side effects: increments self._wins_on_play,
            self._games_on_play, self._wins_on_draw,
            self._games_on_draw in place, adding to whatever was
            already accumulated from prior calls.
        Exceptions: none expected from well-formed input.
        """
        for card_column_set in card_columns:
            in_deck = chunk[card_column_set.deck] > 0
            on_play_mask = in_deck & chunk["on_play"]
            on_draw_mask = in_deck & ~chunk["on_play"]
            uuid = card_column_set.nocab_uuid

            games_on_play_this_chunk = int(on_play_mask.sum())
            if games_on_play_this_chunk > 0:
                wins_on_play_this_chunk = int((on_play_mask & chunk["won"]).sum())
                self._games_on_play[uuid] = (
                    self._games_on_play.get(uuid, 0) + games_on_play_this_chunk
                )
                self._wins_on_play[uuid] = (
                    self._wins_on_play.get(uuid, 0) + wins_on_play_this_chunk
                )

            games_on_draw_this_chunk = int(on_draw_mask.sum())
            if games_on_draw_this_chunk > 0:
                wins_on_draw_this_chunk = int((on_draw_mask & chunk["won"]).sum())
                self._games_on_draw[uuid] = (
                    self._games_on_draw.get(uuid, 0) + games_on_draw_this_chunk
                )
                self._wins_on_draw[uuid] = (
                    self._wins_on_draw.get(uuid, 0) + wins_on_draw_this_chunk
                )

    def finalize(self) -> dict[UUID, MetricResult]:
        """Turn accumulated on-play/on-draw counts into MetricResult rows.

        Only produces a result for a nocab_uuid present in BOTH
        self._games_on_play and self._games_on_draw (i.e. observed at
        least once in each bucket) — see this module's docstring,
        design decision 2, for why a card missing from either bucket
        is omitted entirely rather than assigned a degenerate value.

        value = (self._wins_on_play[uuid] / self._games_on_play[uuid])
            - (self._wins_on_draw[uuid] / self._games_on_draw[uuid]).
        sample_size = min(self._games_on_play[uuid],
            self._games_on_draw[uuid]) — see this module's docstring,
            design decision 1, for why the smaller of the two, not
            their sum.

        Inputs: none (uses all four accumulator dicts).
        Output: one MetricResult per nocab_uuid present in both
            self._games_on_play and self._games_on_draw.
        Side effects: none.
        Exceptions: none expected.
        """
        qualifying_uuids = self._games_on_play.keys() & self._games_on_draw.keys()
        return {
            uuid: MetricResult(
                nocab_uuid=uuid,
                metric_name=self.name,
                value=(self._wins_on_play[uuid] / self._games_on_play[uuid])
                - (self._wins_on_draw[uuid] / self._games_on_draw[uuid]),
                sample_size=min(self._games_on_play[uuid], self._games_on_draw[uuid]),
                expansion=self._expansion,
                format=self._format_code,
            )
            for uuid in qualifying_uuids
        }

    def save_state(self) -> dict:
        """Snapshot all four accumulator dicts as JSON-serializable data.

        UUID keys are stringified (JSON object keys must be strings).

        Inputs: none (uses all four accumulator dicts).
        Output: {"wins_on_play": {<uuid str>: <int>, ...},
            "games_on_play": {...}, "wins_on_draw": {...},
            "games_on_draw": {...}}.
        Side effects: none.
        Exceptions: none.
        """
        return {
            "wins_on_play": {
                str(uuid): count for uuid, count in self._wins_on_play.items()
            },
            "games_on_play": {
                str(uuid): count for uuid, count in self._games_on_play.items()
            },
            "wins_on_draw": {
                str(uuid): count for uuid, count in self._wins_on_draw.items()
            },
            "games_on_draw": {
                str(uuid): count for uuid, count in self._games_on_draw.items()
            },
        }

    def load_state(self, state: dict) -> None:
        """Restore all four accumulator dicts from a save_state() snapshot.

        Inputs:
            state: a dict previously returned by this class's own
                save_state().
        Output: none.
        Side effects: overwrites all four accumulator dicts.
        Exceptions: raises if state isn't shaped as save_state() would
            have produced.
        """
        self._wins_on_play = {
            UUID(uuid_str): count for uuid_str, count in state["wins_on_play"].items()
        }
        self._games_on_play = {
            UUID(uuid_str): count for uuid_str, count in state["games_on_play"].items()
        }
        self._wins_on_draw = {
            UUID(uuid_str): count for uuid_str, count in state["wins_on_draw"].items()
        }
        self._games_on_draw = {
            UUID(uuid_str): count for uuid_str, count in state["games_on_draw"].items()
        }
