"""Template Method base for streaming metrics: one player's final deck
in one run -> one label on that run, one output row per (run, player).

The sts2_runs mirror of ../sts_gg/deck_label_metric.py, with two
differences:
    - Input is a parsed Sts2Run (run_record.py), not a raw dict, so a
      subclass reads a typed field instead of a raw key.
    - No deck is written anywhere. Every (run, player) deck already sits
      in the published Slay the Spire 2 deck box
      (data/final/decks/slay_the_spire_2.db) under the uuid its deck box
      extraction stage minted, and PlayerRun.deck_uuid is that uuid. So
      the output's deck_uuid points into the published box, and the
      metric file is flagged requires_deck_box so a dojo checks that box
      was built from the same CardBinder version.

A subclass sets LABEL_COLUMN / LABEL_TYPE (by reference to its sts_gg
counterpart's, so that counterpart's dojo wrapper reads this file too:
see src/training/dojo_catalog.py's sts2_runs keys), DEFAULT_OUTPUT_PATH,
and implements _label_for(); a None label writes no row.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.sts2_runs.run_record import PlayerRun, Sts2Run
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId

LabelValue = int | bool | str


class DeckLabelMetric(ABC):
    """(run, player) final deck -> (run_id, deck_uuid, LABEL_COLUMN).

    Satisfies Metric[Sts2Run] (../metric.py) structurally.
    """

    LABEL_COLUMN: ClassVar[str]
    LABEL_TYPE: ClassVar[pa.DataType]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self, card_binder: CardBinder, output_path: Path | None = None
    ) -> None:
        """
        Inputs:
            card_binder: the Slay the Spire 2 binder; only its version is
                read (stamped into the output).
            output_path: overrides DEFAULT_OUTPUT_PATH.
        Output: none (constructor).
        Side effects: creates output_path's parent directory; opens
            output_path for writing (truncating it) until finalize().
        Exceptions: whatever ParquetBuilder raises opening output_path.
        """
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        schema = schema_with_version_metadata(
            pa.schema(
                [
                    ("run_id", pa.string()),
                    ("deck_uuid", pa.string()),
                    (self.LABEL_COLUMN, self.LABEL_TYPE),
                ]
            ),
            MetricVersionMetadata(
                game=GameId.SLAY_THE_SPIRE_2,
                card_binder_version=card_binder.version_for(GameId.SLAY_THE_SPIRE_2),
                requires_deck_box=True,
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(self._output_path, schema)

    def accumulate(self, run: Sts2Run) -> None:
        """Buffer one row per player of run whose label is not None.

        Inputs: run (Sts2Run), already filtered by the scanner (see
            scanner.py's scored_run()).
        Output: none.
        Side effects: buffers rows in the open ParquetBuilder.
        Exceptions: whatever _label_for() or the writer raises.

        Example:
            >>> metric = WinMetric(binder, scratch_path)
            >>> metric.accumulate(run)
            >>> metric.finalize()
        """
        for player in run.players:
            label = self._label_for(run, player)
            if label is None:
                continue
            self._writer.write_row(
                {
                    "run_id": run.run_id,
                    "deck_uuid": str(player.deck_uuid),
                    self.LABEL_COLUMN: label,
                }
            )

    def finalize(self) -> Path:
        """Flush and close the writer. Idempotent.

        Inputs: none. Output: the output path.
        Side effects: writes the remaining rows and the parquet footer.
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/sts2_runs/win.parquet')
        """
        self._writer.close()
        return self._output_path

    @abstractmethod
    def _label_for(self, run: Sts2Run, player: PlayerRun) -> LabelValue | None:
        """This (run, player)'s label, or None to write no row.

        Inputs: run (Sts2Run), player (one of run.players).
        Output: a value of LABEL_TYPE's Python type, or None.
        Side effects: none. Exceptions: none expected.
        """
        raise NotImplementedError
