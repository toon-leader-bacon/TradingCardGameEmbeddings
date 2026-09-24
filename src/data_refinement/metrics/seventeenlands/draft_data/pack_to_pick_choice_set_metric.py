"""PackToPickChoiceSetMetric - BRAINSTORM.md's multi-card metric #4
(Pack-to-Pick Choice Set): for every row, the full pack option set plus
which one was picked - a direct multi-group "options vs. choice"
contrastive/ranking example, streamed one output row per input row.

Streaming, not accumulation (see ../../metric.py's module docstring for
the distinction) - one row already carries a complete example, no
cross-row state needed, same shape as sts_gg/deck_label_metric.py's
DeckLabelMetric (open a ParquetWriter in __init__, write one row per
accumulate() call, finalize() only closes it) - except this metric
writes a list-of-options column plus a single pick column, not a
deck_uuid reference, since a pack's option set isn't a GenericDeck and
has no reason to be deduplicated through a DeckBox.

See plans/draft_data_metrics.md's "Open questions" #2 on why this and
pool_conditioned_pick_metric.py's PoolConditionedPickMetric are kept as
independent classes rather than sharing a base, despite their close
similarity.
"""

from pathlib import Path
from typing import Iterable
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pack_pool_columns import (
    DraftCardColumns,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/draft_data/pack_to_pick_choice_set.parquet"
)

_OUTPUT_SCHEMA = pa.schema(
    [
        ("draft_id", pa.string()),
        ("pack_number", pa.int64()),
        ("pick_number", pa.int64()),
        ("pack_option_uuids", pa.list_(pa.string())),
        ("pick_uuid", pa.string()),  # nullable - see _output_row()
    ]
)


class PackToPickChoiceSetMetric:
    """One row's full pack option set -> which option was picked.

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
            card_binder: registry to match this CSV's pack_card_<name>
                column suffixes (and each row's pick cell value)
                against - assumed already fully populated for
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
            existing file) via a pyarrow.parquet.ParquetWriter held
            open for the lifetime of this instance - callers MUST call
            finalize() when done, or the file is left incomplete.
        Exceptions: whatever pyarrow.parquet.ParquetWriter raises on
            failure to open output_path for writing.
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
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Convert one draft_data row into a single output row and
        write it immediately.

        Inputs:
            row: one draft_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: writes exactly one row to the open ParquetWriter.
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = PackToPickChoiceSetMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        pack_option_uuids = self._draft_columns.present_uuids(
            row, self._draft_columns.pack_columns
        )
        pick_uuid = self._draft_columns.uuid_for_name(row["pick"])

        # TODO: memory/perf - same issue as
        # pool_conditioned_pick_metric.py's accumulate() (see its TODO
        # comment for the full writeup): one write_table() call per
        # CSV row means one parquet row group per row, and
        # ParquetWriter's per-row-group metadata grows across the
        # whole run. Observed killing this on MSH.PremierDraft.csv
        # (2026-09-23/24) - see src/training/TODO.md section C.
        output_row = self._output_row(row, pack_option_uuids, pick_uuid)
        self._writer.write_table(output_row)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        A true no-op relative to data - every row this instance will
        ever write was already written by accumulate(). Idempotent: a
        second call is a no-op.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/draft_data/pack_to_pick_choice_set.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _output_row(
        self, row: dict, pack_option_uuids: list[UUID], pick_uuid: UUID | None
    ) -> pa.Table:
        """Build one single-row pa.Table matching _OUTPUT_SCHEMA.

        Private helper - single consumer is accumulate().

        Inputs:
            row: the same row accumulate() received.
            pack_option_uuids: already-matched pack options for this
                row.
            pick_uuid: this row's already-matched pick, or None if
                unmatched (written as a null pick_uuid - see
                sts_gg/deck_label_metrics.py's KilledByMetric for the
                same nullable-column convention; this metric does not
                drop or skip a row over an unmatched pick).
        Output: a one-row pa.Table matching _OUTPUT_SCHEMA:
            draft_id/pack_number/pick_number read straight off row,
            pack_option_uuids/pick_uuid stringified from the matched
            uuids given.
        Side effects: none.
        Exceptions: none expected.
        """
        return pa.Table.from_pydict(
            {
                "draft_id": [row["draft_id"]],
                "pack_number": [row["pack_number"]],
                "pick_number": [row["pick_number"]],
                "pack_option_uuids": [[str(uuid) for uuid in pack_option_uuids]],
                "pick_uuid": [str(pick_uuid) if pick_uuid is not None else None],
            },
            schema=self._output_schema,
        )
