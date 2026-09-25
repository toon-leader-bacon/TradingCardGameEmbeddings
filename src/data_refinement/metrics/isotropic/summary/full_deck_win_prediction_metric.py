"""Streaming metric: BRAINSTORM.md multi-card #2 - one eligible
player's final deck -> rank == 1.

Shape mirrors ../../sts_gg/deck_label_metric.py's DeckLabelMetric (resolve
a final deck, hash it, write it into a shared DeckBox, pull one scalar
label off the same row, write one output row) closely enough that a
future second isotropic per-player-deck-label metric should probably
fold this into an isotropic DeckLabelMetric-style Template Method base
- not done here since this is currently the only concrete consumer of
that exact sequence (BRAINSTORM.md's other per-deck ideas -
deck_card_mask_metric.py's masking shape,
deck_card_set_copy_count_metric.py's per-member-label shape - are
structurally different, not near-duplicates of this one).
"""

from pathlib import Path
from typing import ClassVar

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    deck_for_player,
    eligible_player_entries,
)
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId


class FullDeckWinPredictionMetric:
    """One eligible player's final deck -> rank == 1, one row per
    eligible player, written as soon as accumulate() sees it.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/full_deck_win_prediction.parquet"
    )

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve isotropic card names
                against - must already have dominiontabs' cards
                ingested (this class never writes to it).
            deck_box: the metrics-private DeckBox every deck this
                metric sees is written into - shared with any other
                metric in the same scan pass that also takes a
                DeckBox, so identical final decks dedupe against each
                other. Never the published data/final/decks/ box.
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
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_schema = schema_with_version_metadata(
            pa.schema(
                [
                    ("deck_uuid", pa.string()),
                    ("won", pa.bool_()),
                ]
            ),
            MetricVersionMetadata(
                game=GameId.DOMINION,
                card_binder_version=card_binder.version_for(GameId.DOMINION),
                requires_deck_box=True,
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(self._output_path, self._output_schema)

    def accumulate(self, row: dict) -> None:
        """Convert every eligible player in one game into a single
        (deck_uuid, won) output row each, buffering it for writing.

        Inputs:
            row: one parsed Flavor A summary row, carrying "players"
                (see row_utils.eligible_player_entries()).
        Output: none.
        Side effects: buffers one row into the open ParquetBuilder per
            eligible player. Writes each such player's final deck into
            self._deck_box via create_if_absent(). Emits one
            logging.error() per card name that fails to resolve.
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = FullDeckWinPredictionMetric(card_binder, deck_box)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        for player in eligible_player_entries(row):
            deck = self._deck_for_player(player)
            self._deck_box.create_if_absent(deck)
            self._writer.write_row(
                {
                    "deck_uuid": str(deck.nocab_uuid),
                    "won": player["rank"] == 1,
                }
            )

    def finalize(self) -> Path:
        """Flush any buffered rows and close the underlying writer.

        Does NOT save self._deck_box - that's the calling driver's own
        responsibility, since the box is shared across metrics.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder opened in __init__, if
            not already closed (flushing any rows still buffered).
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/isotropic/full_deck_win_prediction.parquet')
        """
        self._writer.close()
        return self._output_path

    def _deck_for_player(self, player_entry: dict) -> GenericDeck:
        """Build the GenericDeck for one eligible player's end.deck.

        Private helper - single consumer is accumulate(). Delegates to
        row_utils.deck_for_player() - the shared per-player deck
        builder now centralized there, since this file,
        deck_pair_winner_metric.py, and multiplayer_placement_metric.py
        all reached this identical logic independently (the "third
        consumer" trigger this method's own skeleton docstring flagged).

        Inputs:
            player_entry: one players[] entry with a real "end" block.
        Output: a GenericDeck - see row_utils.deck_for_player()'s own
            docstring for the full contract.
        Side effects: none (the caller writes it into deck_box).
        Exceptions: none.
        """
        return deck_for_player(self._card_binder, player_entry)
