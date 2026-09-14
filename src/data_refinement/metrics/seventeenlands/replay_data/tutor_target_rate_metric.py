"""TutorTargetRateMetric (replay-level) -
plans/replay_data_metrics.md's "Tutor Target Rate (replay-level)":
P(card in user_turn_N_cards_tutored for some N | card in deck_<name>).

Distinct class from game_data.tutor_target_rate_metric.TutorTargetRateMetric
- same name reused deliberately, since each lives in its own
source-specific package (the same convention every repeated concept
name across draft_data/game_data already follows). User-only by
construction: there is no oppo_turn_N_cards_tutored column at all
(opponent tutoring is a hidden-library action, confirmed by the
original BRAINSTORM.md pass) - unlike DiscardRateMetric/CastRateMetric,
this isn't a scope choice, it's the only column that exists.

Standalone, same shape as cast_rate_metric.py's CastRateMetric/
discard_rate_metric.py's DiscardRateMetric - not shared with either,
see cast_rate_metric.py's module docstring for why.
"""

from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/replay_data/tutor_target_rate.parquet"
)


class TutorTargetRateMetric:
    """Card -> P(tutored at least once in the game | card in
    deck_<name>).

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
                column suffixes and per-turn cards_tutored Arena-ID
                cells against - assumed already fully populated for
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
        self._total_count: dict[UUID, int] = {}
        self._hit_count: dict[UUID, int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally this game's deck-present cards' (hit, total) toward
        this metric's running per-card state.

        Inputs:
            row: one replay_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: for every card in
            self._replay_columns.present_uuids(row, deck_columns),
            increments self._total_count[card]; for the subset also
            matched out of any user_turn_N_cards_tutored cell on this
            row (any N in user_turn_numbers), increments
            self._hit_count[card].
        Exceptions: implementation-defined (expected: none for a
            well-formed row).

        Example:
            >>> metric = TutorTargetRateMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        deck_uuids = self._replay_columns.present_uuids(
            row, self._replay_columns.deck_columns
        )
        if not deck_uuids:
            return

        tutored_uuids = self._tutored_uuids_for_row(row)
        for card_uuid in deck_uuids:
            self._total_count[card_uuid] = self._total_count.get(card_uuid, 0) + 1
            if card_uuid in tutored_uuids:
                self._hit_count[card_uuid] = self._hit_count.get(card_uuid, 0) + 1

    def finalize(self) -> Path:
        """Compute every seen card's tutor target rate and write one
        row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, tutor_target_rate: float,
            sample_count: int).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/replay_data/tutor_target_rate.parquet')
        """
        result = [self._rate_row(card_uuid) for card_uuid in self._total_count]

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    def _tutored_uuids_for_row(self, row: dict) -> set[UUID]:
        """Every card matched out of any user_turn_N_cards_tutored
        cell on this row, across every N in
        self._replay_columns.user_turn_numbers.

        Private helper - single consumer is accumulate().

        Inputs:
            row: one replay_data CSV row, dict-like.
        Output: the union of every matched card across every scanned
            user half-turn's cards_tutored cell.
        Side effects: none.
        Exceptions: none expected.
        """
        card_uuids: set[UUID] = set()
        for turn in self._replay_columns.user_turn_numbers:
            card_uuids.update(
                self._replay_columns.arena_uuids(
                    row[ReplayCardColumns.turn_column("user", turn, "cards_tutored")]
                )
            )
        return card_uuids

    def _rate_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._total_count.
        Output: a dict with keys "nocab_uuid" (str),
            "tutor_target_rate" (float, self._hit_count.get(card_uuid,
            0) / self._total_count[card_uuid]), "sample_count" (int,
            self._total_count[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        total = self._total_count[card_uuid]
        return {
            "nocab_uuid": str(card_uuid),
            "tutor_target_rate": self._hit_count.get(card_uuid, 0) / total,
            "sample_count": total,
        }
