"""Template Method base for accumulation metrics: per card, the average
of one value over every final-deck copy of that card in every run.

The sts2_runs mirror of ../sts_gg/card_average_metric.py, generalized
one step: the value comes from _value_for(run, player, slot), so it can
depend on the copy itself, not only on the run. That one hook covers
every per-card metric sts_gg has:
    - a run-level average (relics, turns, win, ...): ignores the slot;
    - CardUpgradeRateMetric: the copy's own upgraded flag;
    - CardWinRateAtAct2Metric: the run's win, but only for copies in the
      deck before act 2 started (None for the rest, and for runs that
      never reached act 2).
A None value skips that copy. A rate is the average of a 0/1 indicator.

Copies whose card has no spire_codex alias (CardSlot.card_uuid None, the
Unknown sentinel in the deck box) are never tallied, so the sentinel
never gets a per-card row. Every copy is its own sample, as in sts_gg.

A subclass sets LABEL_COLUMN (its sts_gg counterpart's, by reference,
so that counterpart's dojo wrapper reads this file),
DEFAULT_OUTPUT_PATH, and implements _value_for().
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts2_runs.run_record import (
    CardSlot,
    PlayerRun,
    Sts2Run,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId


class CardAverageMetric(ABC):
    """Card -> mean of _value_for() over its copies, plus sample_count.

    Satisfies Metric[Sts2Run] (../metric.py) structurally.
    """

    LABEL_COLUMN: ClassVar[str]
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
        Side effects: none (no I/O until finalize()).
        Exceptions: none.
        """
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._version_metadata = MetricVersionMetadata(
            game=GameId.SLAY_THE_SPIRE_2,
            card_binder_version=card_binder.version_for(GameId.SLAY_THE_SPIRE_2),
        )
        self._value_sum: dict[UUID, float] = {}
        self._sample_count: dict[UUID, int] = {}

    def accumulate(self, run: Sts2Run) -> None:
        """Tally every aliased card copy of every player of run.

        Inputs: run (Sts2Run), already filtered by the scanner.
        Output: none.
        Side effects: updates the in-memory per-card sums and counts.
        Exceptions: whatever _value_for() raises.

        Example:
            >>> metric = CardWinRateMetric(binder, scratch_path)
            >>> metric.accumulate(run)
            >>> metric.finalize()
        """
        for player in run.players:
            for slot in player.deck:
                if slot.card_uuid is None:
                    continue
                value = self._value_for(run, player, slot)
                if value is None:
                    continue
                card_uuid = slot.card_uuid
                self._value_sum[card_uuid] = self._value_sum.get(card_uuid, 0.0) + value
                self._sample_count[card_uuid] = self._sample_count.get(card_uuid, 0) + 1

    def finalize(self) -> Path:
        """Write one row per tallied card: nocab_uuid (str), LABEL_COLUMN
        (float, the mean), sample_count (int).

        Inputs: none. Output: the output path.
        Side effects: creates the parent directory; writes the parquet
            (with the CardBinder version stamped in).
        Exceptions: whatever the parquet write raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/sts2_runs/card_win_rate.parquet')
        """
        rows = [
            {
                "nocab_uuid": str(card_uuid),
                self.LABEL_COLUMN: self._value_sum[card_uuid] / count,
                "sample_count": count,
            }
            for card_uuid, count in self._sample_count.items()
        ]
        frame = pd.DataFrame(
            rows, columns=["nocab_uuid", self.LABEL_COLUMN, "sample_count"]
        )
        write_dataframe_with_version_metadata(
            frame, self._output_path, self._version_metadata
        )
        return self._output_path

    @abstractmethod
    def _value_for(
        self, run: Sts2Run, player: PlayerRun, slot: CardSlot
    ) -> float | None:
        """This copy's value, or None to leave it out.

        Inputs: run, player (one of run.players), slot (one of
            player.deck, with a card_uuid).
        Output: float | None (a bool or int is fine: it is summed).
        Side effects: none. Exceptions: none expected.
        """
        raise NotImplementedError
