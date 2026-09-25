"""Streaming metric: BRAINSTORM.md multi-card #22 - given the *set* of
distinct cards present in one player's final deck (counts stripped),
predict each one's actual count.

Structurally close to kingdom_member_label_metric.py's
KingdomMemberLabelMetric (one row per (group, member) pair, group
identity minted via DeckBox) but NOT built on it: that base's group is
always the KINGDOM (board.supply, unresolved-count card names known
before any deck is built), whereas this metric's group is a
PLAYER'S OWN FINAL DECK with its counts deliberately erased down to a
plain set - a genuinely different group-construction step, even though
the per-member output row shape coincides. Flagged (like several other
BRAINSTORM.md entries) as a candidate for a shared base if a third
"iterate a card group, emit one row per member" consumer shows up
targeting genuinely the same group-construction step as one of these
two; forcing it now would guess at that future shape.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    card_uuid_for_name,
    eligible_player_entries,
)
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class DeckCardSetCopyCountMetric:
    """(deck_set_uuid, card_uuid, count), one row per (eligible
    player's distinct-card set, card in that set) pair.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/deck_card_set_copy_count.parquet"
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
            deck_box: the metrics-private DeckBox every distinct-card
                SET this metric sees is written into - note this box
                stores sets (counts stripped), not real final decks;
                its entries are NOT interchangeable with e.g.
                full_deck_win_prediction_metric.py's real-deck entries
                even when they happen to share a deck_box instance
                across a scan pass (a set and a multiset over the same
                cards hash to different deck_uuids by construction, so
                they can never collide - see hash_utils.py).
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
                    ("deck_set_uuid", pa.string()),
                    ("card_uuid", pa.string()),
                    ("count", pa.int64()),
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
        """Write one (deck_set_uuid, card_uuid, count) row per distinct
        card, for every eligible player in this game.

        Inputs:
            row: one parsed Flavor A summary row, carrying "players"
                (see row_utils.eligible_player_entries()).
        Output: none.
        Side effects: buffers one row into the open ParquetBuilder per
            (eligible player, distinct card in that player's end.deck)
            pair. Writes each player's distinct-card set into
            self._deck_box via create_if_absent(). Emits one
            logging.error() per card name that fails to resolve.
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = DeckCardSetCopyCountMetric(card_binder, deck_box)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        for player in eligible_player_entries(row):
            end_deck = player["end"]["deck"]
            deck = self._distinct_card_set_deck(end_deck)
            self._deck_box.create_if_absent(deck)

            for card_name, count in end_deck.items():
                card_uuid = card_uuid_for_name(self._card_binder, card_name)
                if card_uuid is None:
                    _logger.error(
                        "DeckCardSetCopyCountMetric: unresolved card name "
                        "%r - excluding it from this deck's output rows",
                        card_name,
                    )
                    continue
                self._writer.write_row(
                    {
                        "deck_set_uuid": str(deck.nocab_uuid),
                        "card_uuid": str(card_uuid),
                        "count": count,
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
            PosixPath('data/metrics/isotropic/deck_card_set_copy_count.parquet')
        """
        self._writer.close()
        return self._output_path

    def _distinct_card_set_deck(self, end_deck: dict[str, int]) -> GenericDeck:
        """Build the GenericDeck for one player's distinct-card set
        (counts stripped to exactly one copy per distinct card).

        Private helper - single consumer is accumulate().

        Inputs:
            end_deck: one player's end.deck dict (name -> copy count).
        Output: a GenericDeck over end_deck's resolved KEYS only (never
            weighted by count), source_game=GameId.DOMINION.
            provenance is left as its default (None) - this is a
            private, metrics-only DeckBox entry, not a canonical
            ingested deck, mirroring ../../sts_gg/deck_label_metric.py's
            own GenericDeck construction (which likewise omits
            provenance entirely).
        Side effects: none (the caller writes it into deck_box).
        Exceptions: none.
        """
        card_nocab_uuids: list[UUID] = []
        for card_name in end_deck:
            card_uuid = card_uuid_for_name(self._card_binder, card_name)
            if card_uuid is not None:
                card_nocab_uuids.append(card_uuid)

        deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
        return GenericDeck(
            nocab_uuid=deck_uuid,
            source_game=GameId.DOMINION,
            name="isotropic distinct-card set",
            card_nocab_uuids=card_nocab_uuids,
        )
