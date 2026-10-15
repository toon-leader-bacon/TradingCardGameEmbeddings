"""Accumulation metric: BRAINSTORM.md single-card #3, as corrected -
P(card vetoed | card offered), where a game's offered cards are
board.supply plus vetoed.

WHY NOT P(card in vetoed | card in board.supply): a vetoed card is banned
before the kingdom is dealt, so it never appears in board.supply (checked
on 18,158 rows of games-20130301: the two lists never overlap). That
first formulation was 0.0 for every card by construction.

ONLY GAMES WITH VETOES: about 40% of rows carry a "vetoed" list, almost
always two cards (one per player), so a row without one most likely had
vetoing switched off rather than two players who chose not to veto.
Those rows are skipped rather than counted as "offered, not vetoed".

The per-card rate is a CardMeanTally (../../card_mean_tally.py) of a 0/1
"vetoed" indicator, the same tally ../games/pile_exhaustion_rate_metric.py
and ../games/opening_buy_rate_metric.py use for their membership rates.
Each of the three still selects its own rows and hits, and keeps its own
_card_uuid_or_log(), rather than sharing a Template Method base.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    is_natural_kingdom,
    kingdom_card_names,
    vetoed_card_names,
    card_uuid_for_name,
)
from src.data_refinement.metrics.card_mean_tally import CardMeanTally
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class VetoRateMetric:
    """P(card vetoed | card offered), per card, over natural-kingdom
    games that had at least one veto (see module docstring).

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    Restricted to natural (unconstrained) kingdoms only - see
    row_utils.is_natural_kingdom()'s docstring for why a
    generator-constrained kingdom's veto behavior isn't a fair sample
    of ordinary play.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/veto_rate.parquet"
    )
    LABEL_COLUMN: ClassVar[str] = "veto_rate"

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
        self._tally = CardMeanTally()

    def accumulate(self, row: dict) -> None:
        """Tally every offered card (board.supply plus vetoed) of a
        natural-kingdom row with vetoes toward this metric's running
        per-card veto rate.

        Inputs:
            row: one parsed Flavor A summary row (see BRAINSTORM.md),
                carrying at least "board" (with "supply") and,
                optionally, "vetoed" (a list of card names).
        Output: none.
        Side effects: adds to self._tally once per distinct offered
            card, for natural kingdoms with a
            non-empty "vetoed" list only (other rows are skipped
            entirely). Emits one logging.error() per offered card name
            that fails to resolve.
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = VetoRateMetric(card_binder)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        # Skip generator-constrained kingdoms entirely - not a fair
        # sample of ordinary veto behavior (see class docstring).
        if not is_natural_kingdom(row):
            return

        # No vetoes most likely means vetoing was off (module docstring)
        vetoed_names = set(vetoed_card_names(row))
        if not vetoed_names:
            return

        # Tally every offered card as one sample, a "hit" iff vetoed
        offered_names = sorted(set(kingdom_card_names(row)) | vetoed_names)
        for card_name in offered_names:
            card_uuid = self._card_uuid_or_log(card_name)
            if card_uuid is None:
                continue
            self._tally.add(card_uuid, 1.0 if card_name in vetoed_names else 0.0)

    def finalize(self) -> Path:
        """Compute every seen card's veto rate and write one row per
        card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, veto_rate: float, sample_count:
            int - one row per card seen at least once).
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/isotropic/veto_rate.parquet')
        """
        write_dataframe_with_version_metadata(
            self._tally.mean_frame(self.LABEL_COLUMN),
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
            card_name: one offered (board.supply or vetoed) card name.
        Output: the matching nocab_uuid, or None if unresolved.
        Side effects: emits one logging.error() call when unresolved.
        Exceptions: none.
        """
        card_uuid = card_uuid_for_name(self._card_binder, card_name)
        if card_uuid is None:
            _logger.error(
                "VetoRateMetric: unresolved card name %r - excluding it "
                "from this row's tally",
                card_name,
            )
        return card_uuid
