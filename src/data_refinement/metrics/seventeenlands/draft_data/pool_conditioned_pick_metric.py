"""PoolConditionedPickMetric - BRAINSTORM.md's multi-group metric #5
(Pool-Conditioned Pick Prediction): given the pool so far AND the
current pack options, which pack option gets picked - richer than
pack_to_pick_choice_set_metric.py's PackToPickChoiceSetMetric since it
additionally conditions on the drafter's already-committed pool.

Same streaming shape as PackToPickChoiceSetMetric (one row in, one row
out, no cross-row state) - see that module's docstring for the
DeckLabelMetric-style constructor/finalize() shape this reuses, and see
plans/draft_data_metrics.md's "Open questions" #2 for why these two
classes are kept independent rather than sharing a base.

pool_<name> columns are read as-is (already excluding this row's own
pick - see plans/draft_data_metrics.md's "Data facts" section), so no
extra "exclude the current pick" logic is needed here.
"""

from pathlib import Path
from typing import Iterable
from uuid import UUID

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.seventeenlands.draft_data.pack_pool_columns import (
    DraftCardColumns,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/draft_data/pool_conditioned_pick.parquet"
)

_OUTPUT_SCHEMA = pa.schema(
    [
        ("draft_id", pa.string()),
        ("pack_number", pa.int64()),
        ("pick_number", pa.int64()),
        ("pool_uuids", pa.list_(pa.string())),
        ("pack_option_uuids", pa.list_(pa.string())),
        ("pick_uuid", pa.string()),  # nullable - see _output_row()
    ]
)


class PoolConditionedPickMetric:
    """One row's pool-so-far + pack options -> which option was picked.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to match this CSV's pack_card_<name>/
                pool_<name> column suffixes (and each row's pick cell
                value) against - assumed already fully populated for
                source_game. Never queried directly by this class -
                only through the DraftCardColumns this constructor
                builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own DraftCardColumns.
            source_game: which game's cards header/pick names are
                matched against.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file) via a ParquetBuilder held open for
            the lifetime of this instance - callers MUST call
            finalize() when done, or the file is left incomplete.
        Exceptions: whatever ParquetBuilder raises on failure
            to open output_path for writing.
        """
        self._draft_columns = DraftCardColumns.from_header(
            header, card_binder, source_game
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_schema = schema_with_version_metadata(
            _OUTPUT_SCHEMA,
            MetricVersionMetadata(
                game=source_game,
                card_binder_version=card_binder.version_for(source_game),
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(self._output_path, self._output_schema)

    def accumulate(self, row: dict) -> None:
        """Convert one draft_data row into a single output row and
        buffer it for writing.

        Inputs:
            row: one draft_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: buffers exactly one row into the open
            ParquetBuilder (flushed to disk automatically once
            its batch size is reached, or by finalize()).
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = PoolConditionedPickMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        pool_uuids = self._draft_columns.present_uuids(
            row, self._draft_columns.pool_columns
        )
        pack_option_uuids = self._draft_columns.present_uuids(
            row, self._draft_columns.pack_columns
        )
        pick_uuid = self._draft_columns.uuid_for_name(row["pick"])

        output_row = self._output_row(row, pool_uuids, pack_option_uuids, pick_uuid)
        self._writer.write_row(output_row)

    def finalize(self) -> Path:
        """Flush any buffered rows and close the underlying writer.

        A true no-op relative to data - every row this instance will
        ever write was already buffered by accumulate(). Idempotent: a
        second call is a no-op.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder opened in
            __init__, if not already closed (flushing any rows still
            buffered).
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/draft_data/pool_conditioned_pick.parquet')
        """
        self._writer.close()
        return self._output_path

    def _output_row(
        self,
        row: dict,
        pool_uuids: list[UUID],
        pack_option_uuids: list[UUID],
        pick_uuid: UUID | None,
    ) -> dict:
        """Build one output row dict matching _OUTPUT_SCHEMA's columns.

        Private helper - single consumer is accumulate().

        Inputs:
            row: the same row accumulate() received.
            pool_uuids: already-matched pool-so-far for this row.
            pack_option_uuids: already-matched pack options for this
                row.
            pick_uuid: this row's already-matched pick, or None if
                unmatched (written as a null pick_uuid - same
                convention as
                pack_to_pick_choice_set_metric.py's _output_row()).
        Output: a dict keyed by every _OUTPUT_SCHEMA column name:
            draft_id/pack_number/pick_number read straight off row,
            pool_uuids/pack_option_uuids/pick_uuid stringified from the
            matched uuids given.
        Side effects: none.
        Exceptions: none expected.
        """
        return {
            "draft_id": row["draft_id"],
            "pack_number": row["pack_number"],
            "pick_number": row["pick_number"],
            "pool_uuids": [str(uuid) for uuid in pool_uuids],
            "pack_option_uuids": [str(uuid) for uuid in pack_option_uuids],
            "pick_uuid": str(pick_uuid) if pick_uuid is not None else None,
        }
