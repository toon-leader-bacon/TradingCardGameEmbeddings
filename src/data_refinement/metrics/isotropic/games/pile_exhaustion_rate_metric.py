"""Accumulation metric: BRAINSTORM.md's "More Flavor B-only candidates"
section, single-card #24 - P(card's pile is named as exhausted at game
end | card in kingdom_card_names).

Flavor-A-impossible (BRAINSTORM.md's own framing): Flavor A's JSON
summary rows have no notion of which supply piles ran out, only the
winner's final deck - this is the first metric in this project able to
answer "how contested is this pile" at all.

Bespoke, same per-card-membership-rate shape as
opening_buy_rate_metric.py and ../summary/veto_rate_metric.py - see
either file's own module docstring for why this isn't yet a shared
Template Method base.

PLURAL VS. SINGULAR NAMES: header.exhausted_pile_names is English-
pluralized ("Provinces", not "Province" - see header_parser.py's
GameHeader.exhausted_pile_names field comment), while
header.kingdom_card_names is already singular. Membership is therefore
tested by UUID (resolve each exhausted name via
row_utils.pile_card_uuid_for_name(), collect the resulting uuids into a
set, then check each already-resolved kingdom card's uuid against that
set) - never by comparing the two name lists as strings, which would
never match.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.isotropic.games.header_parser import GameHeader
from src.data_refinement.metrics.isotropic.games.row_utils import (
    card_uuid_for_name,
    pile_card_uuid_for_name,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class PileExhaustionRateMetric:
    """P(card's pile exhausted by game end | card in kingdom_card_names),
    per kingdom card, restricted to natural kingdoms.

    Satisfies the Metric[GameHeader] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/pile_exhaustion_rate.parquet"
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
        self._exhausted_count: dict[UUID, int] = {}

    def accumulate(self, header: GameHeader) -> None:
        """Tally every kingdom card in a natural-kingdom game toward
        this metric's running per-card (exhausted_count, total_count)
        pairs.

        Inputs:
            header: one parsed GameHeader (header_parser.py).
        Output: none.
        Side effects: updates self._total_count/_exhausted_count in
            place, once per kingdom card, for natural-kingdom games
            only. Emits one logging.error() per kingdom card name that
            fails to resolve.
        Exceptions: none expected beyond a malformed header.

        Example:
            >>> metric = PileExhaustionRateMetric(card_binder)
            >>> metric.accumulate(header)
            >>> metric.finalize()
        """
        if not header.is_natural_kingdom:
            return

        exhausted_uuids = {
            card_uuid
            for card_uuid in (
                pile_card_uuid_for_name(self._card_binder, name)
                for name in header.exhausted_pile_names
            )
            if card_uuid is not None
        }

        for card_name in header.kingdom_card_names:
            card_uuid = self._card_uuid_or_log(card_name)
            if card_uuid is None:
                continue
            self._total_count[card_uuid] = self._total_count.get(card_uuid, 0) + 1
            if card_uuid in exhausted_uuids:
                self._exhausted_count[card_uuid] = (
                    self._exhausted_count.get(card_uuid, 0) + 1
                )

    def finalize(self) -> Path:
        """Compute every seen card's pile-exhaustion rate and write
        one row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, pile_exhaustion_rate: float,
            sample_count: int).
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/isotropic/pile_exhaustion_rate.parquet')
        """
        result = [
            {
                "nocab_uuid": str(card_uuid),
                "pile_exhaustion_rate": self._exhausted_count.get(card_uuid, 0) / total,
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
                "PileExhaustionRateMetric: unresolved card name %r - "
                "excluding it from this game's tally",
                card_name,
            )
        return card_uuid
