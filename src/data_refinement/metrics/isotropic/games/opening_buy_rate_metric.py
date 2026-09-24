"""Accumulation metric: BRAINSTORM.md's "More Flavor B-only candidates"
section, single-card #23 - P(card in {opening 1st buy, opening 2nd
buy} | card in board.supply).

OVER EVERY PLAYER, NOT JUST THE WINNER: BRAINSTORM.md's own wording
("which kingdom cards get bought turn one AT ALL") doesn't restrict
this to winning openings specifically - it's asking a corpus-wide "is
this a common/played opening" question, not "is this a winning
opening" (that's kingdom_opening_buy_prediction_metric.py's and
opening_buy_outcome_metric.py's job). Tallying every player's opening,
not just winners', is both the more natural reading of that wording
and gives roughly 2x the sample size per natural-kingdom game.

Bespoke, not part of a shared Template Method family - same
per-card-membership-rate shape as ../summary/veto_rate_metric.py's
VetoRateMetric, but a different row type entirely (GameHeader, not a
Flavor A dict) - see that file's own module docstring for why a
second real consumer within the SAME row-type family is this
project's trigger for extracting a shared base, not a look-alike
metric over a different source's row shape.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.isotropic.games.header_parser import GameHeader
from src.data_refinement.metrics.isotropic.games.row_utils import card_uuid_for_name
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class OpeningBuyRateMetric:
    """P(card in {opening 1st buy, opening 2nd buy} | card in
    kingdom_card_names), per kingdom card, tallied across every
    player in every natural-kingdom game.

    Satisfies the Metric[GameHeader] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/opening_buy_rate.parquet"
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
        self._total_count: dict[UUID, int] = {}
        self._opening_count: dict[UUID, int] = {}

    def accumulate(self, header: GameHeader) -> None:
        """Tally every kingdom card in a natural-kingdom game toward
        this metric's running per-card (opening_count, total_count)
        pairs, once per player in the game.

        Inputs:
            header: one parsed GameHeader (header_parser.py).
        Output: none.
        Side effects: updates self._total_count/_opening_count in
            place, once per (player, kingdom card) pair, for natural-
            kingdom games only (header.is_natural_kingdom False skips
            the whole header). Emits one logging.error() per kingdom
            card name that fails to resolve.
        Exceptions: none expected beyond a malformed header.

        Example:
            >>> metric = OpeningBuyRateMetric(card_binder)
            >>> metric.accumulate(header)
            >>> metric.finalize()
        """
        if not header.is_natural_kingdom:
            return

        for player in header.players:
            opening_names = {
                name for name in player.opening_buy_names if name is not None
            }
            for card_name in header.kingdom_card_names:
                card_uuid = self._card_uuid_or_log(card_name)
                if card_uuid is None:
                    continue
                self._total_count[card_uuid] = self._total_count.get(card_uuid, 0) + 1
                if card_name in opening_names:
                    self._opening_count[card_uuid] = (
                        self._opening_count.get(card_uuid, 0) + 1
                    )

    def finalize(self) -> Path:
        """Compute every seen card's opening-buy rate and write one
        row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, opening_buy_rate: float,
            sample_count: int).
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/isotropic/opening_buy_rate.parquet')
        """
        result = [
            {
                "nocab_uuid": str(card_uuid),
                "opening_buy_rate": self._opening_count.get(card_uuid, 0) / total,
                "sample_count": total,
            }
            for card_uuid, total in self._total_count.items()
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

    def _card_uuid_or_log(self, card_name: str) -> UUID | None:
        """Resolve one kingdom card name, logging on failure.

        Private helper - single consumer is accumulate().

        Inputs:
            card_name: one kingdom_card_names entry.
        Output: the matching nocab_uuid, or None if unresolved.
        Side effects: emits one logging.error() call when unresolved.
        Exceptions: none.
        """
        card_uuid = card_uuid_for_name(self._card_binder, card_name)
        if card_uuid is None:
            _logger.error(
                "OpeningBuyRateMetric: unresolved card name %r - excluding "
                "it from this game's tally",
                card_name,
            )
        return card_uuid
