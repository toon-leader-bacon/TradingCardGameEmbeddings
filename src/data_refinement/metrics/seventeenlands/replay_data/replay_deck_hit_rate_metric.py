"""Template Method base for a per-card hit/total rate collapsed to ONE
hit/total pair per GAME via a single deck-membership check (see
replay_data/README.md).

Distinct shape from ReplayTurnEventRateMetric
(replay_turn_event_rate_metric.py): that base walks every (actor, turn)
half-turn on a row and tallies a hit/total pair per per-turn
occurrence. This base instead tallies exactly one hit/total pair per
card per row - every card present in this user's own deck_<name>
columns is one "total" occurrence, and whichever of those cards also
appears in this row's own already-collapsed "hit" set (e.g. the union
of every user_turn_N_creatures_cast/non_creatures_cast cell, computed
once per row by the subclass) is one "hit" occurrence. There is no
per-turn denominator/numerator split here - deck membership (not a
per-turn eligibility condition) is the sole denominator, which is what
distinguishes this group (CastRateMetric/DiscardRateMetric/
TutorTargetRateMetric) from ReplayTurnEventRateMetric's subclasses
(CombatKillInvolvementRateMetric/CombatDamagePushThroughRateMetric).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterable
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


class ReplayDeckHitRateMetric(ABC):
    """Card -> P(row-level hit | card in deck_<name>), one hit/total
    pair tallied per card per row.

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
            card_binder: registry to match this CSV's deck_<name>
                column suffixes and per-turn Arena-ID cells against -
                assumed already fully populated for source_game. Never
                queried directly by this class - only through the
                ReplayCardColumns this constructor builds from it.
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
        self._version_metadata = MetricVersionMetadata(
            game=source_game, card_binder_version=card_binder.version_for(source_game)
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
            matched by _hit_uuids_for_row(row), increments
            self._hit_count[card].
        Exceptions: implementation-defined (expected: none for a
            well-formed row).

        Example:
            >>> metric = SomeReplayDeckHitRateMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        deck_uuids = self._replay_columns.present_uuids(
            row, self._replay_columns.deck_columns
        )
        if not deck_uuids:
            return

        hit_uuids = self._hit_uuids_for_row(row)
        for card_uuid in deck_uuids:
            self._total_count[card_uuid] = self._total_count.get(card_uuid, 0) + 1
            if card_uuid in hit_uuids:
                self._hit_count[card_uuid] = self._hit_count.get(card_uuid, 0) + 1

    def finalize(self) -> Path:
        """Compute every seen card's hit_count/total_count rate and
        write one row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, self.LABEL_COLUMN: float,
            sample_count: int).
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
    def _hit_uuids_for_row(self, row: dict) -> set[UUID]:
        """Every card counted as a "hit" on this row.

        Inputs:
            row: one replay_data CSV row, dict-like - same row
                accumulate() received.
        Output: every card_uuid this row counts as a hit (e.g. the
            union of every matched card across every scanned
            user half-turn's relevant event cell). Empty set if none.
        Side effects: implementation-defined (expected: none - reads
            row via self._replay_columns.arena_uuids()).
        Exceptions: implementation-defined (expected: none for a
            well-formed row).
        """
        raise NotImplementedError
