"""Accumulation metric: BRAINSTORM.md single-card #11 -
E[end.deck[card] | card in end.deck].

The reduced-mean sibling of copies_bought_distribution_metric.py's
CopiesBoughtDistributionMetric - see that module's docstring for why
these are two separate classes rather than one. NOT built on
../../sts_gg/card_average_metric.py's CardAverageMetric: that base
broadcasts one ROW-LEVEL scalar (e.g. a run's win/loss) across every
card in the deck, whereas this metric's value is itself PER-CARD (each
card's own copy count) - a shape CardAverageMetric's
_value_for_run(row) -> single value hook cannot express.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    card_uuid_for_name,
    eligible_player_entries,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class AverageCopiesBoughtMetric:
    """E[end.deck[card] | card in end.deck], per card.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/average_copies_bought.parquet"
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
        self._copies_sum: dict[UUID, int] = {}
        self._deck_count: dict[UUID, int] = {}

    def accumulate(self, row: dict) -> None:
        """Fold every eligible player's end.deck counts into this
        metric's running per-card (copies_sum, deck_count) pairs.

        Inputs:
            row: one parsed Flavor A summary row, carrying "players"
                (see row_utils.eligible_player_entries()).
        Output: none.
        Side effects: updates self._copies_sum/_deck_count in place,
            once per (eligible player, distinct card in that player's
            end.deck) pair. Emits one logging.error() per card name
            that fails to resolve.
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = AverageCopiesBoughtMetric(card_binder)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        for player in eligible_player_entries(row):
            for card_name, count in player["end"]["deck"].items():
                card_uuid = card_uuid_for_name(self._card_binder, card_name)
                if card_uuid is None:
                    _logger.error(
                        "AverageCopiesBoughtMetric: unresolved card name "
                        "%r - excluding it from this deck's tally",
                        card_name,
                    )
                    continue
                self._copies_sum[card_uuid] = self._copies_sum.get(card_uuid, 0) + count
                self._deck_count[card_uuid] = self._deck_count.get(card_uuid, 0) + 1

    def finalize(self) -> Path:
        """Compute every seen card's average copy count and write one
        row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, average_copies_bought: float,
            sample_count: int).
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/isotropic/average_copies_bought.parquet')
        """
        result = [
            {
                "nocab_uuid": str(card_uuid),
                "average_copies_bought": self._copies_sum[card_uuid] / count,
                "sample_count": count,
            }
            for card_uuid, count in self._deck_count.items()
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
