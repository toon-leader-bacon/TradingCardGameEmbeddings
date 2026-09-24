"""Template Method base for streaming metrics whose training input is
one game_data row's constructed deck (deck_<name>, referenced by
deck_uuid, never embedded - mirroring sts_gg/deck_label_metric.py's own
DeckBox convention) paired with a single scalar label already present
on that same row - see plans/game_data_metrics.md's Component overview
#7.

Three of round 1's multi-card metrics (game_deck_label_metrics.py's
DeckWinPredictionMetric, DeckGameLengthPredictionMetric,
DeckRankTierPredictionMetric) share this exact sequence of steps -
match the deck's cards via GameCardColumns.deck_columns, hash them via
deck_uuid_from_cards(), write the deck into a private DeckBox, pull one
scalar off the row, write one output row - and differ only in which
field of the row that scalar comes from and what type/column name it's
written under. That's a Template Method (PATTERNS.md): this class owns
every shared step; a subclass only fixes
LABEL_COLUMN/LABEL_TYPE/DEFAULT_OUTPUT_PATH and implements
_label_for_row(). Mirrors sts_gg/deck_label_metric.py's DeckLabelMetric
relationship to its own eleven subclasses, one level over onto a
different raw source.

on_play_win_rate_sensitivity_by_deck_metric.py's
OnPlayWinRateSensitivityByDeckMetric needs the exact same deck
matching + hashing + DeckBox-write steps, but is accumulation, not
streaming (its label needs cross-row aggregation, so it can't fit this
class's one-row-per-input-row shape) - it duplicates those steps
directly rather than sharing a base across the streaming/accumulation
split, the same call draft_data's
pack_to_pick_choice_set_metric.py/pool_conditioned_pick_metric.py
already made for a similar near-duplicate pair (see that container's
README "Open questions" precedent).

deck_box IS REQUIRED (no default) here, unlike
sts_gg/card_average_metric.py's CardAverageMetric (which accepts an
always-unused deck_box purely for cross-class signature consistency
within that container) - every subclass of this class genuinely needs
one, and every metric in this container that has no use for a deck box
(game_card_average_metric.py, on_play_win_rate_delta_metric.py,
tutor_target_rate_metric.py) simply doesn't declare the parameter at
all, following draft_data's leaner precedent instead.

PER-GAME IDENTIFIER: game_data has no single unique-id column - this
container's settled composite is (draft_id: str, match_number: int,
game_number: int), read directly off each row as three separate output
columns, mirroring draft_data's own draft_id/pack_number/pick_number
convention rather than one joined string.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Iterable
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.seventeenlands.game_data.game_card_columns import (
    GameCardColumns,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId


class GameDeckLabelMetric(ABC):
    """One game's constructed deck -> (deck_uuid, label), one row per
    game, written as soon as accumulate() sees it.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    LABEL_COLUMN: ClassVar[str]
    LABEL_TYPE: ClassVar[pa.DataType]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to match this CSV's deck_<name>
                column suffixes against - assumed already fully
                populated for source_game. Never queried directly by
                this class - only through the GameCardColumns this
                constructor builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own GameCardColumns.
            source_game: which game's cards header names are matched
                against.
            deck_box: the metrics-private DeckBox every deck this
                metric sees is written into - shared with any other
                metric in the same scan pass that also takes a
                deck_box, so identical decks dedupe against each other.
                Never the published deck_box/ box.
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
        self._source_game = source_game
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_schema = schema_with_version_metadata(
            pa.schema(
                [
                    ("draft_id", pa.string()),
                    ("match_number", pa.int64()),
                    ("game_number", pa.int64()),
                    ("deck_uuid", pa.string()),
                    (self.LABEL_COLUMN, self.LABEL_TYPE),
                ]
            ),
            MetricVersionMetadata(
                game=source_game,
                card_binder_version=card_binder.version_for(source_game),
                requires_deck_box=True,
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Convert one game_data row into a single output row and write
        it immediately.

        Inputs:
            row: one game_data CSV row, dict-like - carrying at least
                draft_id/match_number/game_number, this row's
                deck_<name> columns, plus whatever field
                _label_for_row() reads.
        Output: none.
        Side effects: writes exactly one row to the open ParquetWriter.
            Writes this row's deck into self._deck_box via
            create_if_absent() (a no-op if an identical deck was
            already written by this or another metric sharing the same
            box).
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = SomeGameDeckLabelMetric(card_binder, header, GameId.MTG, deck_box)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        label = self._label_for_row(row)

        card_nocab_uuids = self._game_columns.present_uuids(
            row, self._game_columns.deck_columns
        )
        deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
        self._deck_box.create_if_absent(
            self._deck_for_row(row, deck_uuid, card_nocab_uuids)
        )

        output_row = self._output_row(row, deck_uuid, label)
        self._writer.write_table(output_row)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        A true no-op relative to data - every row this instance will
        ever write was already written by accumulate(). Idempotent: a
        second call is a no-op. Does NOT save self._deck_box - that's
        the calling driver's own responsibility, since the box is
        shared across metrics and only the driver knows when every
        metric sharing it is done.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/some_label.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    @abstractmethod
    def _label_for_row(self, row: dict) -> Any:
        """Pull this metric's one scalar label off a raw game_data row.

        The only step of accumulate()'s sequence a subclass overrides -
        every other step (deck identification, hashing, DeckBox write,
        output row write) is shared in this base class.

        Inputs:
            row: one game_data CSV row, dict-like - same row
                accumulate() received.
        Output: the label value for self.LABEL_COLUMN - must already
            match self.LABEL_TYPE's Python representation.
        Side effects: implementation-defined (expected: none - a plain
            field read).
        Exceptions: implementation-defined (expected: raises if row is
            missing the field this subclass reads).
        """
        raise NotImplementedError

    def _deck_for_row(
        self, row: dict, deck_uuid: UUID, card_nocab_uuids: list[UUID]
    ) -> GenericDeck:
        """Build the GenericDeck this row's deck_<name> multiset
        represents, for writing into self._deck_box.

        Private helper - single consumer is accumulate().

        Inputs:
            row: one game_data CSV row, dict-like.
            deck_uuid: this deck's already-hashed identity
                (deck_uuid_from_cards(card_nocab_uuids)).
            card_nocab_uuids: this row's already-matched deck_<name>
                present cards.
        Output: a GenericDeck (src/schema/card.py) named e.g.
            f"game_data {row['draft_id']}/{row['match_number']}/
            {row['game_number']} deck", with source_game=self._source_game
            and card_nocab_uuids=card_nocab_uuids.
        Side effects: none.
        Exceptions: none expected.
        """
        return GenericDeck(
            nocab_uuid=deck_uuid,
            source_game=self._source_game,
            name=(
                f"game_data {row['draft_id']}/{row['match_number']}/"
                f"{row['game_number']} deck"
            ),
            card_nocab_uuids=card_nocab_uuids,
        )

    def _output_row(self, row: dict, deck_uuid: UUID, label: Any) -> pa.Table:
        """Build one single-row pa.Table matching self._output_schema.

        Private helper - single consumer is accumulate().

        Inputs:
            row: the same row accumulate() received.
            deck_uuid: this row's already-hashed deck identity.
            label: this row's already-computed label
                (_label_for_row(row)).
        Output: a one-row pa.Table matching self._output_schema:
            draft_id/match_number/game_number read straight off row,
            deck_uuid stringified, label under self.LABEL_COLUMN.
        Side effects: none.
        Exceptions: none expected.
        """
        return pa.Table.from_pydict(
            {
                "draft_id": [row["draft_id"]],
                "match_number": [row["match_number"]],
                "game_number": [row["game_number"]],
                "deck_uuid": [str(deck_uuid)],
                self.LABEL_COLUMN: [label],
            },
            schema=self._output_schema,
        )
