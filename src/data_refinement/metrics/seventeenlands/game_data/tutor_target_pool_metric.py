"""TutorTargetPoolMetric - BRAINSTORM.md's multi-card metric "Predict
Tutor Targets Given Deck": given one game's full draft pool
(deck_<name> union sideboard_<name>), label each pool card with whether
it appears in tutored_<name> that game.

NEW SHAPE - FAN-OUT STREAMING: every other streaming metric in this
codebase (draft_data/pack_to_pick_choice_set_metric.py,
draft_data/pool_conditioned_pick_metric.py,
game_deck_label_metric.py's GameDeckLabelMetric family) writes exactly
one output row per accumulate() call. This metric writes one row per
POOL CARD, so one accumulate() call may write zero or more rows
(pa.Table.from_pydict() naturally accepts a multi-row dict, so this
needs no new writer mechanics beyond building a wider dict before one
write_table() call). See plans/game_data_metrics.md's Component
overview #10 and open question #2 for this shape's own rationale and
the alternative (a single parallel-list row per game) flagged there for
human review.

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
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
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
            existing file) via a pyarrow.parquet.ParquetWriter held
            open for the lifetime of this instance - callers MUST call
            finalize() when done, or the file is left incomplete.
        Exceptions: whatever pyarrow.parquet.ParquetWriter raises on
            failure to open output_path for writing.
        """
        self._game_columns = GameCardColumns.from_header(
            header, card_binder, source_game
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, _OUTPUT_SCHEMA)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Convert one game_data row into zero or more output rows (one
        per pool card) and write them immediately.

        Inputs:
            row: one game_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: writes one row per distinct pool card (deck_<name>
            union sideboard_<name> present on this row) to the open
            ParquetWriter - zero rows if the pool is empty.
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

        output_rows = self._output_rows(row, pool_uuids, tutored_uuids)
        self._writer.write_table(output_rows)

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
            PosixPath('data/metrics/seventeenlands/game_data/tutor_target_pool.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
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
    ) -> pa.Table:
        """Build one multi-row pa.Table matching _OUTPUT_SCHEMA, one row
        per pool card.

        Private helper - single consumer is accumulate().

        Inputs:
            row: the same row accumulate() received.
            pool_uuids: this row's already-computed full pool
                (_pool_uuids(row)).
            tutored_uuids: this row's already-matched tutored_<name>
                present cards, as a set for membership checks.
        Output: a pa.Table matching _OUTPUT_SCHEMA with len(pool_uuids)
            rows: draft_id/match_number/game_number repeated per row,
            pool_card_uuid one entry per pool_uuids member,
            tutored = pool_card_uuid in tutored_uuids.
        Side effects: none.
        Exceptions: none expected.
        """
        row_count = len(pool_uuids)
        return pa.Table.from_pydict(
            {
                "draft_id": [row["draft_id"]] * row_count,
                "match_number": [row["match_number"]] * row_count,
                "game_number": [row["game_number"]] * row_count,
                "pool_card_uuid": [str(card_uuid) for card_uuid in pool_uuids],
                "tutored": [card_uuid in tutored_uuids for card_uuid in pool_uuids],
            },
            schema=_OUTPUT_SCHEMA,
        )
