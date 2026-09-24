"""Accumulation metric: BRAINSTORM.md single-card #7 - a signed float,
the average winner turns over games where a card is in board.supply,
minus the corpus-wide average winner turns.

Bespoke rather than built on a shared "per-kingdom-card average" base:
the baseline subtraction (a SECOND, corpus-wide running average, kept
alongside the per-card ones and only combined at finalize() time) is
unique to this metric among BRAINSTORM.md's current shortlist - forcing
it into a generic average base now would mean designing that base's
baseline-subtraction hook against a single real consumer, the same
premature-abstraction call ../../sts_gg/card_win_rate_at_act2_metric.py's
own module docstring makes for its act-2 conditioning.

Restricted to natural (unconstrained) kingdoms only, same reasoning as
veto_rate_metric.py - see row_utils.is_natural_kingdom().
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    card_uuid_for_name,
    is_natural_kingdom,
    kingdom_card_names,
    winner_entry,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class TurnCountAssociationMetric:
    """avg(winner turns | card in board.supply) - avg(winner turns
    over the whole natural-kingdom corpus), per kingdom card.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/turn_count_association.parquet"
    )

    def __init__(
        self,
        card_binder: CardBinder,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve isotropic card names
                against - must already have dominiontabs' cards
                ingested (this class never writes to it).
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none - no I/O happens until finalize() is
            called.
        Exceptions: none.
        """
        self._card_binder = card_binder
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._turns_sum: dict[UUID, int] = {}
        self._game_count: dict[UUID, int] = {}
        self._corpus_turns_sum = 0
        self._corpus_game_count = 0

    def accumulate(self, row: dict) -> None:
        """Fold one natural-kingdom game's winner turn count into both
        this metric's per-card tallies and its corpus-wide baseline
        tally.

        Inputs:
            row: one parsed Flavor A summary row, carrying "board"
                (with "supply") and "players".
        Output: none.
        Side effects: updates self._turns_sum/_game_count per kingdom
            card, and self._corpus_turns_sum/_corpus_game_count once,
            for natural-kingdom games with a resolvable winner (see
            row_utils.winner_entry()) - a game with no winner
            (row_utils.winner_entry() returns None, e.g. every player
            resigned) contributes nothing to either tally. Emits one
            logging.error() per kingdom card name that fails to
            resolve.
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = TurnCountAssociationMetric(card_binder)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        if not is_natural_kingdom(row):
            return

        winner = winner_entry(row)
        if winner is None:
            return
        turns = winner["turns"]

        self._corpus_turns_sum += turns
        self._corpus_game_count += 1

        for card_name in kingdom_card_names(row):
            card_uuid = card_uuid_for_name(self._card_binder, card_name)
            if card_uuid is None:
                _logger.error(
                    "TurnCountAssociationMetric: unresolved card name %r - "
                    "excluding it from this game's tally",
                    card_name,
                )
                continue
            self._turns_sum[card_uuid] = self._turns_sum.get(card_uuid, 0) + turns
            self._game_count[card_uuid] = self._game_count.get(card_uuid, 0) + 1

    def finalize(self) -> Path:
        """Compute every seen card's baseline-adjusted turn-count
        association and write one row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, turn_count_delta: float (signed -
            positive means this card's kingdoms run longer than
            average), sample_count: int).
        Exceptions: whatever pyarrow.parquet.write_table raises, or
            ZeroDivisionError if this instance's accumulate() was never
            called with any natural-kingdom, winner-resolvable game
            (self._corpus_game_count == 0) - a real "no baseline to
            compare against" failure a caller should not paper over.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/isotropic/turn_count_association.parquet')
        """
        baseline = self._corpus_turns_sum / self._corpus_game_count

        result = [
            {
                "nocab_uuid": str(card_uuid),
                "turn_count_delta": self._turns_sum[card_uuid] / count - baseline,
                "sample_count": count,
            }
            for card_uuid, count in self._game_count.items()
        ]

        write_dataframe_with_version_metadata(
            pd.DataFrame(result),
            self._output_path,
            MetricVersionMetadata(
                game=GameId.DOMINION,
                card_binder_version=self._card_binder.version_for(GameId.DOMINION),
            ),
        )
        return self._output_path
