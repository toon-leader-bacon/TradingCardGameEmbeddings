"""Template Method base for accumulation metrics that tally, per card
seen present (count > 0) in some game_data column set, the running
average of one scalar derived from that same row - see
plans/game_data_metrics.md's Component overview #2.

Three of round 1's single-card metrics (game_card_average_metrics.py's
WinRateWhenInDeckMetric, OpeningHandWinRateMetric, DrawnWinRateMetric)
share this exact accumulate() sequence - look up every present card in
one column set, tally value_sum/total_count - and differ only in which
GameCardColumns column set counts as "present" (_present_card_uuids())
and what value gets averaged (_value_for_row()). That's a Template
Method (PATTERNS.md): this class owns every shared step; a subclass
only fixes LABEL_COLUMN/DEFAULT_OUTPUT_PATH and implements
_present_card_uuids()/_value_for_row(). Mirrors
sts_gg/card_average_metric.py's CardAverageMetric relationship to its
own nine subclasses, one level over onto a different raw source.

game_length_association_metric.py's GameLengthAssociationMetric also
subclasses this base, reusing accumulate()/_present_card_uuids()/
_value_for_row() but needing one more piece of state (a format-wide
baseline) this base's own accumulate() doesn't track. Rather than
letting that subclass override accumulate() itself (a break in this
class's own Template Method contract - PATTERNS.md: the base owns the
skeleton, a subclass only varies designated steps), accumulate() below
ends by calling _extra_accumulate(row), an optional hook (no-op by
default) that exists for exactly this - see that module's own
docstring for how it's used.

CARD_BINDER + HEADER, NOT A SHARED GameCardColumns: each metric takes
(card_binder, header, source_game) directly - mirroring draft_data's
PackCardTallyMetric and sts_gg's CardAverageMetric/DeckLabelMetric
always taking card_binder in their own constructors - and builds its
own GameCardColumns internally (GameCardColumns.from_header()) rather
than receiving an already-built one from a shared driver.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId


class GameCardAverageMetric(ABC):
    """Per-card running average of one row-derived scalar, across every
    game a card was present in (per some GameCardColumns column set).

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
            card_binder: registry to match this CSV's
                opening_hand_<name>/drawn_<name>/tutored_<name>/
                deck_<name>/sideboard_<name> column suffixes against -
                assumed already fully populated for source_game. Never
                queried directly by this class - only through the
                GameCardColumns this constructor builds from it.
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
        self._version_metadata = MetricVersionMetadata(
            game=source_game, card_binder_version=card_binder.version_for(source_game)
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._value_sum: dict[UUID, float] = {}
        self._total_count: dict[UUID, int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every present card this row qualifies toward this
        metric's running per-card (value_sum, total_count), then run
        this subclass's optional extra bookkeeping.

        Inputs:
            row: one game_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: updates self._value_sum/_total_count in place,
            once per card_uuid returned by
            self._present_card_uuids(row); calls
            self._extra_accumulate(row) once, unconditionally, as the
            last step.
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract
            for how a raised exception here is actually handled during
            a real scan).

        Example:
            >>> metric = SomeGameCardAverageMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        value = self._value_for_row(row)

        # Tally every card this subclass's column set has present on
        # this row.
        for card_uuid in self._present_card_uuids(row):
            self._value_sum[card_uuid] = self._value_sum.get(card_uuid, 0.0) + value
            self._total_count[card_uuid] = self._total_count.get(card_uuid, 0) + 1

        # Hook for a subclass needing extra, non-card-keyed bookkeeping
        # every row (see GameLengthAssociationMetric) - no-op by
        # default.
        self._extra_accumulate(row)

    def finalize(self) -> Path:
        """Compute every seen card's average and write one row per card
        to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, self.LABEL_COLUMN: float,
            sample_count: int - one row per card seen at least once).
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/some_average.parquet')
        """
        result: list[dict] = [
            self._average_row(card_uuid) for card_uuid in self._total_count
        ]

        write_dataframe_with_version_metadata(
            pd.DataFrame(result), self._output_path, self._version_metadata
        )
        return self._output_path

    @abstractmethod
    def _present_card_uuids(self, row: dict) -> list[UUID]:
        """Which cards count as "present" on this row, for this
        subclass's own column set.

        One of the two steps a subclass overrides - together with
        _value_for_row(), defines what this metric measures.

        Inputs:
            row: one game_data CSV row, dict-like.
        Output: every card_uuid this subclass's chosen column set
            (e.g. self._game_columns.deck_columns) has present (count >
            0) on row - typically
            self._game_columns.present_uuids(row, <column set>).
        Side effects: none expected.
        Exceptions: implementation-defined.
        """
        raise NotImplementedError

    @abstractmethod
    def _value_for_row(self, row: dict) -> float:
        """The scalar to average for every card _present_card_uuids()
        returns on this row.

        The other step a subclass overrides.

        Inputs:
            row: one game_data CSV row, dict-like - same row
                accumulate() received.
        Output: a value convertible to float (e.g. 1.0/0.0 for
            row["won"]).
        Side effects: implementation-defined (expected: none - a plain
            field read).
        Exceptions: implementation-defined.
        """
        raise NotImplementedError

    def _extra_accumulate(self, row: dict) -> None:
        """Optional extra per-row bookkeeping a subclass needs beyond
        the shared per-card tally above.

        No-op by default. Overridden by
        game_length_association_metric.py's GameLengthAssociationMetric
        to track a format-wide baseline that isn't gated on any card's
        presence - see that module's own docstring. This hook exists so
        such a subclass never has to override accumulate() itself (see
        this module's own docstring).

        Inputs:
            row: one game_data CSV row, dict-like - same row
                accumulate() received.
        Output: none.
        Side effects: none by default; implementation-defined for an
            overriding subclass.
        Exceptions: implementation-defined.
        """
        # No-op by default - see class docstring.
        return

    def _average_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._total_count.
        Output: a dict with keys "nocab_uuid" (str), self.LABEL_COLUMN
            (float, self._value_sum[card_uuid] divided by
            self._total_count[card_uuid]), and "sample_count" (int,
            self._total_count[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        total = self._total_count[card_uuid]
        return {
            "nocab_uuid": str(card_uuid),
            self.LABEL_COLUMN: self._value_sum[card_uuid] / total,
            "sample_count": total,
        }
