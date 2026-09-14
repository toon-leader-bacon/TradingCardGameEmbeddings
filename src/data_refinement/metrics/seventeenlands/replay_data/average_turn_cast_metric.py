"""AverageTurnCastMetric - plans/replay_data_metrics.md's "Average Turn
Cast": for a card, the average turn number it's cast on, across every
game it's cast in at all.

Standalone - NOT a ReplayTurnEventRateMetric. Its per-occurrence value
is the turn number itself, not a hit/miss boolean, so it needs its own
(turn_sum, occurrence_count) running tally per card rather than
(hit_count, total_count). Both sides' occurrences count - no deck-
membership conditioning needed (a card cast by either player is valid
signal here, unlike CastRateMetric's user-deck-only conditioning).
"""

from pathlib import Path
from typing import ClassVar, Iterable, Literal
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/replay_data/average_turn_cast.parquet"
)


class AverageTurnCastMetric:
    """Card -> average turn number cast on, across every (actor, turn)
    occurrence in every game scanned.

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
            card_binder: registry to match this CSV's per-turn
                creatures_cast/non_creatures_cast Arena-ID cells
                against - assumed already fully populated for
                source_game. Never queried directly by this class -
                only through the ReplayCardColumns this constructor
                builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own ReplayCardColumns.
            source_game: which game's cards header names are matched
                against.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none beyond building this instance's own
            ReplayCardColumns from card_binder/header.
        Exceptions: none.
        """
        self._replay_columns = ReplayCardColumns.from_header(
            header, card_binder, source_game
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._turn_sum: dict[UUID, int] = {}
        self._occurrence_count: dict[UUID, int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every (actor, turn) cast occurrence's turn number
        toward this metric's running per-card state.

        Inputs:
            row: one replay_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: for each actor in ("user", "oppo"), for each turn
            in that actor's own turn-number range, matches every card
            in that half-turn's creatures_cast union
            non_creatures_cast and updates self._turn_sum/
            self._occurrence_count in place, once per occurrence.
        Exceptions: implementation-defined (expected: none for a
            well-formed row).

        Example:
            >>> metric = AverageTurnCastMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        for actor in ReplayCardColumns.ACTORS:
            turn_numbers = (
                self._replay_columns.user_turn_numbers
                if actor == "user"
                else self._replay_columns.oppo_turn_numbers
            )
            for turn in turn_numbers:
                for card_uuid in self._cast_uuids_for_turn(row, actor, turn):
                    self._turn_sum[card_uuid] = self._turn_sum.get(card_uuid, 0) + turn
                    self._occurrence_count[card_uuid] = (
                        self._occurrence_count.get(card_uuid, 0) + 1
                    )

    def finalize(self) -> Path:
        """Compute every seen card's average cast turn and write one
        row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, average_turn_cast: float,
            sample_count: int).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/replay_data/average_turn_cast.parquet')
        """
        result = [self._cast_row(card_uuid) for card_uuid in self._occurrence_count]

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    def _cast_uuids_for_turn(
        self, row: dict, actor: Literal["user", "oppo"], turn: int
    ) -> list[UUID]:
        """Every card matched out of this half-turn's creatures_cast
        union non_creatures_cast cell.

        Private helper - single consumer is accumulate().

        Inputs:
            row: one replay_data CSV row, dict-like.
            actor: which half-turn - "user" or "oppo".
            turn: that actor's own turn-number counter.
        Output: every matched card in creatures_cast union
            non_creatures_cast for this half-turn. Empty list if
            neither cell names a card.
        Side effects: none.
        Exceptions: none expected.
        """
        creatures = self._replay_columns.arena_uuids(
            row[ReplayCardColumns.turn_column(actor, turn, "creatures_cast")]
        )
        non_creatures = self._replay_columns.arena_uuids(
            row[ReplayCardColumns.turn_column(actor, turn, "non_creatures_cast")]
        )
        return creatures + non_creatures

    def _cast_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._occurrence_count.
        Output: a dict with keys "nocab_uuid" (str), "average_turn_cast"
            (float, self._turn_sum[card_uuid] /
            self._occurrence_count[card_uuid]), "sample_count" (int,
            self._occurrence_count[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        count = self._occurrence_count[card_uuid]
        return {
            "nocab_uuid": str(card_uuid),
            "average_turn_cast": self._turn_sum[card_uuid] / count,
            "sample_count": count,
        }
