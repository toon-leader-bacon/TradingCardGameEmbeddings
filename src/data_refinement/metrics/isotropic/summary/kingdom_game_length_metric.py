"""Streaming metric: BRAINSTORM.md multi-card #4 - given the kingdom,
predict the eventual winner's turn count.

Restricted to natural (unconstrained) kingdoms with a resolvable
winner, same reasoning as turn_count_association_metric.py - this is
that metric's multi-card counterpart, trained on the same underlying
fact (winner turns given the kingdom) but as one joint per-kingdom
label rather than per-card marginal associations.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    card_uuid_for_name,
    is_natural_kingdom,
    kingdom_card_names,
    winner_entry,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class KingdomGameLengthMetric:
    """(kingdom_uuid, winner_turns), one row per natural-kingdom game
    with a resolvable winner, written as soon as accumulate() sees it.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/kingdom_game_length.parquet"
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
            deck_box: the metrics-private DeckBox every kingdom this
                metric sees is written into as its own card group
                (mirrors kingdom_veto_prediction_metric.py's candidate
                pool convention, minus the vetoed cards).
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
        self._card_binder = card_binder
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_schema = pa.schema(
            [
                ("kingdom_uuid", pa.string()),
                ("winner_turns", pa.int64()),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Write one (kingdom_uuid, winner_turns) row for one natural-
        kingdom game with a resolvable winner.

        Inputs:
            row: one parsed Flavor A summary row, carrying "board"
                (with "supply") and "players".
        Output: none.
        Side effects: writes exactly one row to the open ParquetWriter
            when the row is a natural kingdom (row_utils.
            is_natural_kingdom()) AND has a resolvable winner
            (row_utils.winner_entry() is not None) - writes nothing
            otherwise. Writes the kingdom into self._deck_box via
            create_if_absent().
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = KingdomGameLengthMetric(card_binder, deck_box)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        if not is_natural_kingdom(row):
            return

        winner = winner_entry(row)
        if winner is None:
            return

        kingdom_deck = self._kingdom_deck(kingdom_card_names(row))
        self._deck_box.create_if_absent(kingdom_deck)

        output_row = pa.Table.from_pydict(
            {
                "kingdom_uuid": [str(kingdom_deck.nocab_uuid)],
                "winner_turns": [winner["turns"]],
            },
            schema=self._output_schema,
        )
        self._writer.write_table(output_row)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        Does NOT save self._deck_box - that's the calling driver's own
        responsibility, since the box is shared across metrics.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/isotropic/kingdom_game_length.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _kingdom_deck(self, kingdom_names: list[str]) -> GenericDeck:
        """Build the GenericDeck for one game's kingdom card group.

        Private helper - single consumer is accumulate().

        Inputs:
            kingdom_names: this game's dealt board.supply.
        Output: a GenericDeck over the resolved kingdom names (a set
            of distinct cards, no meaningful copy counts),
            source_game=GameId.DOMINION. provenance is left as its
            default (None) - this is a private, metrics-only DeckBox
            entry, not a canonical ingested deck, mirroring
            ../../sts_gg/deck_label_metric.py's own GenericDeck
            construction (which likewise omits provenance entirely).
        Side effects: none (the caller writes it into deck_box).
        Exceptions: none.
        """
        card_nocab_uuids: list[UUID] = []
        for card_name in kingdom_names:
            card_uuid = card_uuid_for_name(self._card_binder, card_name)
            if card_uuid is None:
                _logger.error(
                    "KingdomGameLengthMetric: unresolved card name %r - "
                    "excluding it from this kingdom",
                    card_name,
                )
                continue
            card_nocab_uuids.append(card_uuid)

        deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
        return GenericDeck(
            nocab_uuid=deck_uuid,
            source_game=GameId.DOMINION,
            name="isotropic kingdom",
            card_nocab_uuids=card_nocab_uuids,
        )
