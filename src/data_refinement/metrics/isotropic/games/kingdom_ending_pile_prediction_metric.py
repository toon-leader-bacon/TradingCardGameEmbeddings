"""Streaming metric: BRAINSTORM.md's "More Flavor B-only candidates"
section, multi-card #28 - given the kingdom, label each individual
kingdom card with whether its own pile was one of the piles named as
exhausted by game end.

ONE ROW PER (KINGDOM, KINGDOM CARD) PAIR: structurally the games/
analogue of ../summary/kingdom_member_label_metric.py's
KingdomMemberLabelMetric Template Method, but NOT built on (or as) a
shared base - this is currently the only "iterate the kingdom, label
each member" metric in games/, so there's no second real consumer yet
to justify extracting one (mirrors that same file's own restraint
about not sharing across GameHeader vs. Flavor A dict row types).

Restricted to natural kingdoms - see header_parser.py's
GameHeader.is_natural_kingdom.

PLURAL VS. SINGULAR NAMES: same UUID-based membership test as
pile_exhaustion_rate_metric.py's own module docstring describes -
header.exhausted_pile_names is English-pluralized, so membership is
tested by resolving each exhausted name via
row_utils.pile_card_uuid_for_name() into a uuid set, never by
comparing name strings directly.
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
from src.data_refinement.metrics.isotropic.games.header_parser import GameHeader
from src.data_refinement.metrics.isotropic.games.row_utils import (
    card_uuid_for_name,
    pile_card_uuid_for_name,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class KingdomEndingPilePredictionMetric:
    """(kingdom_uuid, card_uuid, exhausted), one row per (natural-
    kingdom game, kingdom card) pair.

    Satisfies the Metric[GameHeader] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/kingdom_ending_pile_prediction.parquet"
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
                metric sees is written into as its own card group.
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
        self._output_schema = schema_with_version_metadata(
            pa.schema(
                [
                    ("kingdom_uuid", pa.string()),
                    ("card_uuid", pa.string()),
                    ("exhausted", pa.bool_()),
                ]
            ),
            MetricVersionMetadata(
                game=GameId.DOMINION,
                card_binder_version=card_binder.version_for(GameId.DOMINION),
                requires_deck_box=True,
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, header: GameHeader) -> None:
        """Write one (kingdom_uuid, card_uuid, exhausted) row per
        kingdom card, for one natural-kingdom game.

        Inputs:
            header: one parsed GameHeader (header_parser.py).
        Output: none.
        Side effects: writes one row to the open ParquetWriter per
            kingdom card, when header is a natural kingdom - writes
            nothing otherwise. Writes the kingdom into self._deck_box
            via create_if_absent(). Emits one logging.error() per
            kingdom card name that fails to resolve.
        Exceptions: none expected beyond a malformed header.

        Example:
            >>> metric = KingdomEndingPilePredictionMetric(card_binder, deck_box)
            >>> metric.accumulate(header)
            >>> metric.finalize()
        """
        if not header.is_natural_kingdom:
            return

        exhausted_uuids = {
            card_uuid
            for card_uuid in (
                pile_card_uuid_for_name(self._card_binder, name)
                for name in header.exhausted_pile_names
            )
            if card_uuid is not None
        }

        kingdom_deck = self._kingdom_deck(header.kingdom_card_names)
        self._deck_box.create_if_absent(kingdom_deck)

        for card_name in header.kingdom_card_names:
            card_uuid = card_uuid_for_name(self._card_binder, card_name)
            if card_uuid is None:
                _logger.error(
                    "KingdomEndingPilePredictionMetric: unresolved kingdom "
                    "card name %r - excluding it from this row's output",
                    card_name,
                )
                continue

            output_row = pa.Table.from_pydict(
                {
                    "kingdom_uuid": [str(kingdom_deck.nocab_uuid)],
                    "card_uuid": [str(card_uuid)],
                    "exhausted": [card_uuid in exhausted_uuids],
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
            PosixPath('data/metrics/isotropic/kingdom_ending_pile_prediction.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _kingdom_deck(self, kingdom_names: tuple[str, ...]) -> GenericDeck:
        """Build the GenericDeck for one game's kingdom card group.

        Private helper - single consumer is accumulate(). Same shape
        as every other "kingdom as a card group" builder in this
        project - see kingdom_opening_buy_prediction_metric.py's own
        _kingdom_deck() docstring for why this isn't shared across
        files.

        Inputs:
            kingdom_names: this game's dealt kingdom_card_names.
        Output: a GenericDeck over the resolved kingdom names -
            unresolved names are skipped, not raised on (mirroring
            card_resolution.card_uuid_for_name()'s per-name contract),
            so this method's own docstring does NOT promise
            len(card_nocab_uuids) == len(kingdom_names).
            source_game=GameId.DOMINION, provenance left at its
            default (None).
        Side effects: none (the caller writes it into deck_box). Emits
            one logging.error() per kingdom card name that fails to
            resolve.
        Exceptions: none.
        """
        card_nocab_uuids: list[UUID] = []
        for card_name in kingdom_names:
            card_uuid = card_uuid_for_name(self._card_binder, card_name)
            if card_uuid is None:
                _logger.error(
                    "KingdomEndingPilePredictionMetric: unresolved kingdom "
                    "card name %r - excluding it from this kingdom",
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
