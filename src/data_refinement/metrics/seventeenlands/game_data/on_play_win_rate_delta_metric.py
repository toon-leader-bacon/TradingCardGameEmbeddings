"""OnPlayWinRateDeltaMetric - BRAINSTORM.md's single-card metric
"On-Play vs. On-Draw Win Rate Delta": per card, P(won | card in deck,
on_play=True) - P(won | card in deck, on_play=False) - a tempo/curve-
sensitivity proxy.

Standalone - does NOT subclass GameCardAverageMetric
(game_card_average_metric.py). That base's shape is a single running
(value_sum, total_count) per card; this metric's tally is two-
dimensional per card, keyed by (card_uuid, on_play), so reusing that
base's one-dimensional shape would be forcing an abstraction over only
superficial similarity - the same call draft_data already made for
PickNumberDecayCurveMetric vs. its siblings, just settled here as
"don't subclass" instead of "subclass and override finalize()" since
even accumulate()'s tally key differs here (see
plans/game_data_metrics.md's Component overview #5).

NULLABLE OUTPUT: if a card was never seen on one side (on_play or
on_draw) across every scanned game, that side's rate is undefined - the
delta for that card is written as None rather than guessed at, the same
convention draft_data/pick_number_decay_curve_metric.py uses for an
unseen pick_number bucket.
"""

from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/on_play_win_rate_delta.parquet"
)


class OnPlayWinRateDeltaMetric:
    """Card -> P(won | in deck, on_play) - P(won | in deck, on_draw).

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to match this CSV's deck_<name>
                column suffixes against - assumed already fully
                populated for source_game. Never queried directly by
                this class - only through the GameCardColumns this
                constructor builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own GameCardColumns.
            source_game: which game's cards header names are matched
                against.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none beyond building this instance's own
            GameCardColumns from card_binder/header - no further I/O
            happens until finalize() is called.
        Exceptions: none.
        """
        self._game_columns = GameCardColumns.from_header(
            header, card_binder, source_game
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._win_count: dict[tuple[UUID, bool], int] = {}
        self._total_count: dict[tuple[UUID, bool], int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every deck-present card's (win_count, total_count),
        keyed by (card_uuid, on_play), toward this metric's running
        state.

        Inputs:
            row: one game_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: updates self._win_count/_total_count in place,
            once per card present (count > 0) in deck_<name> on this
            row.
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = OnPlayWinRateDeltaMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        on_play = bool(row["on_play"])
        won = bool(row["won"])

        # Tally every deck-present card toward its own (card, on_play)
        # key.
        for card_uuid in self._game_columns.present_uuids(
            row, self._game_columns.deck_columns
        ):
            key = (card_uuid, on_play)
            self._total_count[key] = self._total_count.get(key, 0) + 1
            if won:
                self._win_count[key] = self._win_count.get(key, 0) + 1

    def finalize(self) -> Path:
        """Compute every seen card's on-play/on-draw win rate delta and
        write one row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, on_play_win_rate_delta: float |
            None, sample_count: int - one row per card seen at least
            once on either side).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/on_play_win_rate_delta.parquet')
        """
        result: list[dict] = [
            self._delta_row(card_uuid) for card_uuid in self._distinct_cards()
        ]

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    def _distinct_cards(self) -> list[UUID]:
        """Every distinct card_uuid tallied at least once, on either
        side of on_play.

        Private helper - single consumer is finalize().

        Inputs: none (uses accumulated state).
        Output: every distinct key[0] across self._total_count, order
            not guaranteed.
        Side effects: none.
        Exceptions: none.
        """
        return list({card_uuid for card_uuid, _ in self._total_count})

    def _win_rate(self, card_uuid: UUID, on_play: bool) -> float | None:
        """This card's win rate on one side of on_play, or None if it
        was never seen on that side.

        Private helper - single consumer is _delta_row().

        Inputs:
            card_uuid: card to look up.
            on_play: which side to compute the rate for.
        Output: self._win_count[(card_uuid, on_play)] /
            self._total_count[(card_uuid, on_play)], or None if
            self._total_count has no entry for that key.
        Side effects: none.
        Exceptions: none.
        """
        key = (card_uuid, on_play)
        if key not in self._total_count:
            return None
        return self._win_count.get(key, 0) / self._total_count[key]

    def _delta_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a card seen in self._total_count on at least one
                side.
        Output: a dict with keys "nocab_uuid" (str),
            "on_play_win_rate_delta" (float | None - None if either
            side's _win_rate() is None), "sample_count" (int, this
            card's total_count summed across both sides).
        Side effects: none.
        Exceptions: none.
        """
        on_play_rate = self._win_rate(card_uuid, True)
        on_draw_rate = self._win_rate(card_uuid, False)

        delta = (
            on_play_rate - on_draw_rate
            if on_play_rate is not None and on_draw_rate is not None
            else None
        )
        sample_count = self._total_count.get(
            (card_uuid, True), 0
        ) + self._total_count.get((card_uuid, False), 0)

        return {
            "nocab_uuid": str(card_uuid),
            "on_play_win_rate_delta": delta,
            "sample_count": sample_count,
        }
