"""Template Method base: BRAINSTORM.md's "New candidates" section
#12/#13 - given the kingdom, label each individual kingdom card with
something derived from the eventual winner's final deck (in it or not,
how many copies).

Two concrete metrics (kingdom_member_label_metrics.py) share this
exact sequence - resolve the kingdom, hash it into a shared group via
DeckBox, iterate every kingdom card, look each one up in the winner's
end.deck, write one row per (kingdom, card) pair - and differ only in
what LABEL_TYPE that per-card lookup produces (a bool for membership, an
int for copy count). Textbook Template Method (PATTERNS.md), the same
shape ../../generic/deck_card_mask_metric.py uses for its own "one row per
targeted card" sequence, but ONE ROW PER KINGDOM MEMBER rather than one
row per row-with-a-single-target - this base is not a
DeckCardMaskMetric subclass because it emits many rows per input row,
not at most one.

Restricted to natural (unconstrained) kingdoms with a resolvable
winner - same reasoning as kingdom_game_length_metric.py.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    card_uuid_for_name,
    is_natural_kingdom,
    kingdom_card_names,
    winner_entry,
)
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class KingdomMemberLabelMetric(ABC):
    """(kingdom_uuid, card_uuid, label), one row per (natural-kingdom
    game with a resolvable winner, kingdom card) pair.

    Structural Metric[dict] (../../metric.py) - every concrete subclass
    satisfies it, since accumulate()/finalize() are defined here and
    inherited unchanged.
    """

    LABEL_COLUMN: ClassVar[str]
    LABEL_TYPE: ClassVar[pa.DataType]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

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
                metric sees is written into as its own card group.
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
                    ("kingdom_uuid", pa.string()),
                    ("card_uuid", pa.string()),
                    (self.LABEL_COLUMN, self.LABEL_TYPE),
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
        """Write one (kingdom_uuid, card_uuid, label) row per kingdom
        card, for one natural-kingdom game with a resolvable winner.

        Inputs:
            row: one parsed Flavor A summary row, carrying "board"
                (with "supply") and "players".
        Output: none.
        Side effects: buffers one row into the open ParquetBuilder per
            kingdom card, when the row is a natural kingdom AND has a
            resolvable winner - writes nothing otherwise (a real "no
            label to derive" outcome, not an error). Writes the
            kingdom into self._deck_box via create_if_absent(). Emits
            one logging.error() per kingdom card name that fails to
            resolve.
        Exceptions: whatever self._label_for_kingdom_card() raises for
            its own malformed input.

        Example:
            >>> metric = WinningDeckMembershipMetric(card_binder, deck_box)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        if not is_natural_kingdom(row):
            return

        winner = winner_entry(row)
        if winner is None:
            return
        winner_end_deck = winner["end"]["deck"]

        kingdom_names = kingdom_card_names(row)
        kingdom_deck = self._kingdom_deck(kingdom_names)
        self._deck_box.create_if_absent(kingdom_deck)

        for card_name in kingdom_names:
            card_uuid = card_uuid_for_name(self._card_binder, card_name)
            if card_uuid is None:
                _logger.error(
                    "%s: unresolved card name %r - excluding it from this "
                    "kingdom's output rows",
                    type(self).__name__,
                    card_name,
                )
                continue

            label = self._label_for_kingdom_card(card_name, winner_end_deck)
            self._writer.write_row(
                {
                    "kingdom_uuid": str(kingdom_deck.nocab_uuid),
                    "card_uuid": str(card_uuid),
                    self.LABEL_COLUMN: label,
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
            PosixPath('data/metrics/isotropic/winning_deck_membership.parquet')
        """
        self._writer.close()
        return self._output_path

    @abstractmethod
    def _label_for_kingdom_card(
        self, card_name: str, winner_end_deck: dict[str, int]
    ) -> Any:
        """The true label for one kingdom card, given the winner's
        final deck counts.

        The only step of accumulate()'s sequence a subclass overrides.

        Inputs:
            card_name: one board.supply entry.
            winner_end_deck: the resolved winner's
                players[i]["end"]["deck"] dict (name -> copy count).
        Output: a value matching self.LABEL_TYPE's Python
            representation (e.g. a plain bool for pa.bool_(), a plain
            int for pa.int64()).
        Side effects: implementation-defined (expected: none - a plain
            dict lookup).
        Exceptions: implementation-defined (expected: none).
        """
        raise NotImplementedError

    def _kingdom_deck(self, kingdom_names: list[str]) -> GenericDeck:
        """Build the GenericDeck for one game's kingdom card group.

        Private helper - single consumer is accumulate(). Identical in
        shape to kingdom_game_length_metric.py's own
        _kingdom_deck() - not shared across the two files, since each
        is otherwise-independent and this is a two-line body once
        implemented; worth revisiting if a third consumer appears.

        Inputs:
            kingdom_names: this game's dealt board.supply.
        Output: a GenericDeck over the resolved kingdom names,
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
            if card_uuid is not None:
                card_nocab_uuids.append(card_uuid)

        deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
        return GenericDeck(
            nocab_uuid=deck_uuid,
            source_game=GameId.DOMINION,
            name="isotropic kingdom",
            card_nocab_uuids=card_nocab_uuids,
        )
