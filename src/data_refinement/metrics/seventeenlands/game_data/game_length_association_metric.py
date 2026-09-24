"""GameLengthAssociationMetric - BRAINSTORM.md's single-card metric
"Game Length Association": for a card, the average num_turns across
every game it was present in deck_<name>, minus the format-wide
average num_turns across every game scanned - a signed
aggro(negative)/control(positive) curve-sensitivity proxy.

Still a GameCardAverageMetric (game_card_average_metric.py) subclass -
reuses accumulate() unchanged (see that class's own docstring for why
its accumulate() ends with an _extra_accumulate(row) hook specifically
so a subclass like this one never has to override accumulate()
itself). This class:

  - overrides _extra_accumulate() to track its own running
    (_global_turn_sum, _global_game_count) - updated unconditionally,
    every row, never gated on any card's presence, since the baseline
    is format-wide, not per-card;
  - overrides finalize() to subtract that baseline from each card's
    own average instead of writing the base class's plain average -
    the same kind of documented Template Method exception
    draft_data/pick_number_decay_curve_metric.py's
    PickNumberDecayCurveMetric already established for a differently-
    shaped finalize() override.

_present_card_uuids()/_value_for_row() are still implemented exactly
like WinRateWhenInDeckMetric's own hooks (deck_<name> presence), except
the averaged value is row["num_turns"], not a win/loss indicator.
"""

from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_average_metric import (
    GameCardAverageMetric,
)
from src.data_refinement.metrics.version_metadata import (
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/game_length_association.parquet"
)


class GameLengthAssociationMetric(GameCardAverageMetric):
    """Card -> (average num_turns when in deck) - (format-wide average
    num_turns)."""

    LABEL_COLUMN: ClassVar[str] = "game_length_association"
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        output_path: Path | None = None,
    ) -> None:
        """Same parameters as GameCardAverageMetric.__init__(). Adds
        this subclass's own running (_global_turn_sum,
        _global_game_count) state, updated by _extra_accumulate()
        below rather than accumulate() itself.

        Inputs: see GameCardAverageMetric.__init__().
        Output: none (constructor).
        Side effects: same as GameCardAverageMetric.__init__(), plus
            initializing self._global_turn_sum/_global_game_count to
            zero.
        Exceptions: none.
        """
        super().__init__(card_binder, header, source_game, output_path)
        self._global_turn_sum = 0.0
        self._global_game_count = 0

    def _present_card_uuids(self, row: dict) -> list[UUID]:
        """See GameCardAverageMetric._present_card_uuids(). Cards
        present (count > 0) in deck_<name> this row."""
        return self._game_columns.present_uuids(row, self._game_columns.deck_columns)

    def _value_for_row(self, row: dict) -> float:
        """See GameCardAverageMetric._value_for_row(). row["num_turns"],
        as a float."""
        return float(row["num_turns"])

    def _extra_accumulate(self, row: dict) -> None:
        """Update this instance's format-wide (_global_turn_sum,
        _global_game_count) baseline, unconditionally, for every row -
        see module docstring.

        Overrides GameCardAverageMetric._extra_accumulate() (the
        designated hook for exactly this kind of extension - see that
        class's own docstring).

        Inputs:
            row: one game_data CSV row, dict-like - same row
                accumulate() received.
        Output: none.
        Side effects: updates self._global_turn_sum/_global_game_count
            in place.
        Exceptions: implementation-defined (expected: none for a
            well-formed row).
        """
        self._global_turn_sum += float(row["num_turns"])
        self._global_game_count += 1

    def finalize(self) -> Path:
        """Compute every seen card's (own average - format-wide
        baseline) and write one row per card to self._output_path.

        Overrides GameCardAverageMetric.finalize() entirely - see
        module docstring for why this metric's output can't reuse the
        shared per-card finalize() as-is.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, game_length_association: float,
            sample_count: int - one row per card seen at least once).
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/game_length_association.parquet')
        """
        # Every seen card's own average, minus the format-wide baseline
        # - see _association_row() for the per-card computation this
        # delegates to (deliberately not GameCardAverageMetric's own
        # _average_row(), which has no baseline to subtract).
        result: list[dict] = [
            self._association_row(card_uuid) for card_uuid in self._total_count
        ]

        write_dataframe_with_version_metadata(
            pd.DataFrame(result), self._output_path, self._version_metadata
        )
        return self._output_path

    def _association_row(self, card_uuid: UUID) -> dict:
        """Build one output row for a single already-tallied card.

        Private helper - single consumer is finalize(). The
        GameLengthAssociationMetric counterpart to
        GameCardAverageMetric._average_row(), except the value written
        is this card's own average minus the format-wide baseline
        rather than the plain average.

        Inputs:
            card_uuid: a key already present in self._total_count.
        Output: a dict with keys "nocab_uuid" (str),
            "game_length_association" (float,
            (self._value_sum[card_uuid] / self._total_count[card_uuid])
            minus (self._global_turn_sum / self._global_game_count)),
            and "sample_count" (int, self._total_count[card_uuid]).
        Side effects: none.
        Exceptions: none.
        """
        total = self._total_count[card_uuid]
        card_average = self._value_sum[card_uuid] / total
        global_average = self._global_turn_sum / self._global_game_count
        return {
            "nocab_uuid": str(card_uuid),
            "game_length_association": card_average - global_average,
            "sample_count": total,
        }
