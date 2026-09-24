"""Template Method base for a per-card hit/total rate tallied across
every per-turn occurrence in a game, rather than once per row - see
plans/replay_data_metrics.md's ReplayTurnEventRateMetric component.

The turn-indexed generalization of game_data's GameCardAverageMetric/
sts_gg's CardAverageMetric shape, one level over: instead of one value
applied uniformly to every card present on a row, this walks every
(actor, turn) half-turn on that same row and tallies a hit/total pair
per card across every occurrence it finds there.
CombatKillInvolvementRateMetric/CombatDamagePushThroughRateMetric
(replay_turn_event_rate_metrics.py) share this exact double-nested
actor x turn loop and differ only in which cards are "eligible" each
half-turn and which of those count as a "hit."

Deliberately NOT shared with CastRateMetric/DiscardRateMetric/
TutorTargetRateMetric (each its own standalone file): those three
collapse to one hit/total pair PER GAME (a single set intersection
against deck-present cards), not one pair per per-turn occurrence -
see plans/replay_data_metrics.md's "why this earns a shared base"
note for the full reasoning distinguishing the two groups.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterable, Literal
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId


class ReplayTurnEventRateMetric(ABC):
    """Per-card hit/total rate across every per-turn occurrence in
    every game scanned.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    LABEL_COLUMN: ClassVar[str]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to match this CSV's deck_<name>/
                sideboard_<name> column suffixes and per-turn Arena-ID
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
            ReplayCardColumns from card_binder/header - no further I/O
            happens until finalize() is called.
        Exceptions: none.
        """
        self._replay_columns = ReplayCardColumns.from_header(
            header, card_binder, source_game
        )
        self._version_metadata = MetricVersionMetadata(
            game=source_game, card_binder_version=card_binder.version_for(source_game)
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._total_count: dict[UUID, int] = {}
        self._hit_count: dict[UUID, int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every (actor, turn) half-turn's denominator/numerator
        cards toward this metric's running per-card state.

        Inputs:
            row: one replay_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: for each actor in ("user", "oppo"), for each turn
            in self._replay_columns.<actor>_turn_numbers, calls
            _denominator_cards_for_turn() and (if non-empty)
            _numerator_cards_for_turn(), updating self._total_count/
            self._hit_count in place.
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = SomeReplayTurnEventRateMetric(card_binder, header, GameId.MTG)
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
                denominator_cards = self._denominator_cards_for_turn(row, actor, turn)
                if not denominator_cards:
                    continue

                for card_uuid in denominator_cards:
                    self._total_count[card_uuid] = (
                        self._total_count.get(card_uuid, 0) + 1
                    )

                hit_cards = self._numerator_cards_for_turn(
                    row, actor, turn, denominator_cards
                )
                for card_uuid in hit_cards:
                    self._hit_count[card_uuid] = self._hit_count.get(card_uuid, 0) + 1

    def finalize(self) -> Path:
        """Compute every seen card's hit_count/total_count rate and
        write one row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, self.LABEL_COLUMN: float,
            sample_count: int - one row per card seen at least once).
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/replay_data/some_rate.parquet')
        """
        result = [self._rate_row(card_uuid) for card_uuid in self._total_count]

        write_dataframe_with_version_metadata(
            pd.DataFrame(result), self._output_path, self._version_metadata
        )
        return self._output_path

    def _rate_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._total_count.
        Output: a dict with keys "nocab_uuid" (str), self.LABEL_COLUMN
            (float, self._hit_count.get(card_uuid, 0) /
            self._total_count[card_uuid]), "sample_count" (int,
            self._total_count[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        total = self._total_count[card_uuid]
        return {
            "nocab_uuid": str(card_uuid),
            self.LABEL_COLUMN: self._hit_count.get(card_uuid, 0) / total,
            "sample_count": total,
        }

    @abstractmethod
    def _denominator_cards_for_turn(
        self, row: dict, actor: Literal["user", "oppo"], turn: int
    ) -> list[UUID]:
        """Cards eligible this (actor, turn) half-turn.

        Inputs:
            row: one replay_data CSV row, dict-like - same row
                accumulate() received.
            actor: which half-turn - "user" or "oppo".
            turn: that actor's own turn-number counter.
        Output: every card_uuid eligible this half-turn (e.g. every
            card that fought - attacked or blocked). Empty list if
            none.
        Side effects: implementation-defined (expected: none - reads
            row via self._replay_columns.arena_uuids()).
        Exceptions: implementation-defined (expected: none for a
            well-formed row).
        """
        raise NotImplementedError

    @abstractmethod
    def _numerator_cards_for_turn(
        self,
        row: dict,
        actor: Literal["user", "oppo"],
        turn: int,
        denominator_cards: list[UUID],
    ) -> set[UUID]:
        """Subset of denominator_cards counted as a hit this
        (actor, turn) half-turn.

        Inputs:
            row: one replay_data CSV row, dict-like.
            actor: which half-turn - "user" or "oppo".
            turn: that actor's own turn-number counter.
            denominator_cards: this same half-turn's already-computed
                _denominator_cards_for_turn() result.
        Output: the subset of denominator_cards counted as a hit this
            half-turn. May be the full set (a turn-wide boolean
            condition applied uniformly) or a genuine per-card subset.
        Side effects: implementation-defined (expected: none).
        Exceptions: implementation-defined (expected: none for a
            well-formed row).
        """
        raise NotImplementedError
