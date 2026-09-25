"""TutorTargetPoolMetric - BRAINSTORM.md's multi-card metric "Predict
Tutor Targets Given Deck": given one game's full draft pool
(deck_<name> union sideboard_<name>), label each pool card with whether
it appears in tutored_<name> that game.

NEW SHAPE - FAN-OUT STREAMING: every other streaming metric in this
codebase (draft_data/pack_to_pick_choice_set_metric.py,
draft_data/pool_conditioned_pick_metric.py,
game_deck_label_metric.py's GameDeckLabelMetric family) writes exactly
one output row per accumulate() call. This metric writes one row per
POOL CARD, so one accumulate() call may write zero or more rows -
built as a list of row dicts, then one ParquetBuilder.write_row() call
per row (ParquetBuilder buffers and batches internally, so fanning out
into several write_row() calls costs nothing extra here). See
plans/game_data_metrics.md's Component overview #10 and open question
#2 for this shape's own rationale and the alternative (a single
parallel-list row per game) flagged there for human review.

NOT A DECK: this metric's identity unit is the per-game
(draft_id, match_number, game_number) triple plus a pool_card_uuid -
NOT a deck_uuid. The pool here is deck_<name> UNION sideboard_<name>, a
different card multiset than any GenericDeck this container mints
elsewhere (game_deck_label_metric.py/
on_play_win_rate_sensitivity_by_deck_metric.py hash only deck_<name>) -
reusing deck_uuid_from_cards() on this wider multiset would conflate
two distinct identities under one hash scheme, so this class takes no
deck_box and never calls that function.
"""

from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/tutor_target_pool.parquet"
)

_OUTPUT_SCHEMA = pa.schema(
    [
        ("draft_id", pa.string()),
        ("match_number", pa.int64()),
        ("game_number", pa.int64()),
        ("pool_card_uuid", pa.string()),
        ("tutored", pa.bool_()),
    ]
)


class TutorTargetPoolMetric:
    """One game's full draft pool -> per-pool-card tutored bool, one
    output row per (game, pool card) pair.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to match this CSV's deck_<name>/
                sideboard_<name>/tutored_<name> column suffixes against
                - assumed already fully populated for source_game.
                Never queried directly by this class - only through the
                GameCardColumns this constructor builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own GameCardColumns.
            source_game: which game's cards header names are matched
                against.
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
        self._game_columns = GameCardColumns.from_header(
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
        """Convert one game_data row into zero or more output rows (one
        per pool card) and buffer them for writing.

        Inputs:
            row: one game_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: buffers one row per distinct pool card (deck_<name>
            union sideboard_<name> present on this row) into the open
            ParquetBuilder (flushed to disk automatically once its
            batch size is reached, or by finalize()) - zero rows if the
            pool is empty.
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = TutorTargetPoolMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        pool_uuids = self._pool_uuids(row)
        if not pool_uuids:
            return

        tutored_uuids = set(
            self._game_columns.present_uuids(row, self._game_columns.tutored_columns)
        )

        for output_row in self._output_rows(row, pool_uuids, tutored_uuids):
            self._writer.write_row(output_row)

    def finalize(self) -> Path:
        """Flush any buffered rows and close the underlying writer.

        A true no-op relative to data - every row this instance will
        ever write was already buffered by accumulate(). Idempotent: a
        second call is a no-op.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder opened in __init__, if
            not already closed (flushing any rows still buffered).
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/tutor_target_pool.parquet')
        """
        self._writer.close()
        return self._output_path

    def _pool_uuids(self, row: dict) -> list[UUID]:
        """This row's full draft pool: deck_<name> union sideboard_<name>
        present cards, deduplicated.

        Private helper - single consumer is accumulate(). See module
        docstring for why GameCardColumns itself doesn't grow a
        pool-specific method for this single consumer's need.

        Inputs:
            row: one game_data CSV row, dict-like.
        Output: every distinct card_uuid present (count > 0) in either
            deck_<name> or sideboard_<name> on this row, order not
            guaranteed.
        Side effects: none.
        Exceptions: none expected.
        """
        deck_uuids = self._game_columns.present_uuids(
            row, self._game_columns.deck_columns
        )
        sideboard_uuids = self._game_columns.present_uuids(
            row, self._game_columns.sideboard_columns
        )
        return list(set(deck_uuids) | set(sideboard_uuids))

    def _output_rows(
        self, row: dict, pool_uuids: list[UUID], tutored_uuids: set[UUID]
    ) -> list[dict]:
        """Build one output row dict per pool card, matching
        _OUTPUT_SCHEMA's columns.

        Private helper - single consumer is accumulate().

        Inputs:
            row: the same row accumulate() received.
            pool_uuids: this row's already-computed full pool
                (_pool_uuids(row)).
            tutored_uuids: this row's already-matched tutored_<name>
                present cards, as a set for membership checks.
        Output: a list of len(pool_uuids) dicts, each keyed by every
            _OUTPUT_SCHEMA column name: draft_id/match_number/
            game_number repeated per row, pool_card_uuid one entry per
            pool_uuids member, tutored = pool_card_uuid in
            tutored_uuids.
        Side effects: none.
        Exceptions: none expected.
        """
        return [
            {
                "draft_id": row["draft_id"],
                "match_number": row["match_number"],
                "game_number": row["game_number"],
                "pool_card_uuid": str(card_uuid),
                "tutored": card_uuid in tutored_uuids,
            }
            for card_uuid in pool_uuids
        ]
