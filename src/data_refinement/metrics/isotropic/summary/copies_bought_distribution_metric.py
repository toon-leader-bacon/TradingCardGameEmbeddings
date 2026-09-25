"""Streaming metric: BRAINSTORM.md single-card #4 - the observed
distribution of end.deck[card] across every game a card appears in.

Streaming, not accumulation: each player's end.deck entry is already a
complete, self-contained sample of "this many copies, in one finished
deck" - there's nothing to aggregate before writing, unlike
average_copies_bought_metric.py's AverageCopiesBoughtMetric (which
tallies the same underlying quantity down to a single per-card mean).
Deliberately kept as a separate metric/file from that one rather than
one class with two output modes - one writes a raw per-observation
sample table for a consumer to bucket/histogram itself, the other
writes an already-reduced scalar; forcing both through one class would
mean a shared accumulate() with a mode flag, which is more machinery
than two independent ~20-line classes.
"""

import logging
from pathlib import Path
from typing import ClassVar

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    card_uuid_for_name,
    eligible_player_entries,
)
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class CopiesBoughtDistributionMetric:
    """One output row per (card, copies-in-one-finished-deck)
    observation - a raw sample table, not a reduced statistic.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/copies_bought_distribution.parquet"
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
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file) via a ParquetBuilder held open for the
            lifetime of this instance - callers MUST call finalize()
            when done, or the file is left incomplete.
        Exceptions: whatever ParquetBuilder raises on failure to open
            output_path for writing.
        """
        self._card_binder = card_binder
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_schema = schema_with_version_metadata(
            pa.schema(
                [
                    ("nocab_uuid", pa.string()),
                    ("copies", pa.int64()),
                ]
            ),
            MetricVersionMetadata(
                game=GameId.DOMINION,
                card_binder_version=card_binder.version_for(GameId.DOMINION),
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(self._output_path, self._output_schema)

    def accumulate(self, row: dict) -> None:
        """Write one (card, copies) row per distinct card in every
        eligible player's end.deck in this game.

        Inputs:
            row: one parsed Flavor A summary row, carrying "players"
                (see row_utils.eligible_player_entries()).
        Output: none.
        Side effects: buffers one row into the open ParquetBuilder per
            (eligible player, distinct card in that player's end.deck)
            pair - one full finished deck contributes one sample per
            distinct card it contains, not per physical copy (the copy
            count itself IS the label, so it must not also multiply
            the sample count). Emits one logging.error() per card name
            that fails to resolve.
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = CopiesBoughtDistributionMetric(card_binder)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        for player in eligible_player_entries(row):
            for card_name, count in player["end"]["deck"].items():
                card_uuid = card_uuid_for_name(self._card_binder, card_name)
                if card_uuid is None:
                    _logger.error(
                        "CopiesBoughtDistributionMetric: unresolved card "
                        "name %r - excluding it from this deck's samples",
                        card_name,
                    )
                    continue
                self._writer.write_row(
                    {
                        "nocab_uuid": str(card_uuid),
                        "copies": count,
                    }
                )

    def finalize(self) -> Path:
        """Flush any buffered rows and close the underlying writer.

        A true no-op relative to data - every row this instance will
        ever write was already buffered by accumulate(). Idempotent.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder opened in __init__, if
            not already closed (flushing any rows still buffered).
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/isotropic/copies_bought_distribution.parquet')
        """
        self._writer.close()
        return self._output_path
