"""Streaming metric: BRAINSTORM.md's "More Flavor B-only candidates"
section, multi-card #29 - given the kingdom, classify how the game
ended: Province exhaustion, Colony exhaustion, or a multi-pile ending.

A coarser, cheaper sibling of kingdom_ending_pile_prediction_metric.py
- just the ending TYPE, not which specific piles. LABEL_VALUES is
fixed here (unlike that file's per-card variable-set label), since the
ending-type vocabulary is small and closed.

MULTI_PILE, NOT "THREE_PILE": a large live sample of 2013-03-15
confirmed every observed multi-pile ending names exactly 3 piles, but
Dominion's own end-game rule is "3 OR MORE piles empty," so this
metric's label vocabulary doesn't bake in "always 3" - see
header_parser.py's GameHeader.exhausted_pile_names docstring for the
same choice made at the parsing layer.
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
    is_single_pile_ending,
    pile_card_uuid_for_name,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_PROVINCE_ENDING = "province"
_COLONY_ENDING = "colony"
_MULTI_PILE_ENDING = "multi_pile"

LABEL_VALUES: tuple[str, ...] = (_PROVINCE_ENDING, _COLONY_ENDING, _MULTI_PILE_ENDING)


class KingdomEndingTypeMetric:
    """(kingdom_uuid, ending_type), one row per natural-kingdom game.

    Satisfies the Metric[GameHeader] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/kingdom_ending_type.parquet"
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
        self._output_schema = pa.schema(
            [
                ("kingdom_uuid", pa.string()),
                ("ending_type", pa.string()),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False
        # Resolved once - compared against by uuid, not by string, since
        # header.exhausted_pile_names is English-pluralized ("Provinces")
        # while a plain card_uuid_for_name() lookup needs the singular
        # name (see _ending_type()'s own docstring).
        self._province_uuid = card_uuid_for_name(card_binder, "Province")
        self._colony_uuid = card_uuid_for_name(card_binder, "Colony")

    def accumulate(self, header: GameHeader) -> None:
        """Write one (kingdom_uuid, ending_type) row for one natural-
        kingdom game.

        Inputs:
            header: one parsed GameHeader (header_parser.py).
        Output: none.
        Side effects: writes exactly one row to the open ParquetWriter
            when header is a natural kingdom - writes nothing
            otherwise. Writes the kingdom into self._deck_box via
            create_if_absent(). Emits one logging.error() per kingdom
            card name that fails to resolve (see _kingdom_deck()).
        Exceptions: none expected beyond a malformed header.

        Example:
            >>> metric = KingdomEndingTypeMetric(card_binder, deck_box)
            >>> metric.accumulate(header)
            >>> metric.finalize()
        """
        if not header.is_natural_kingdom:
            return

        kingdom_deck = self._kingdom_deck(header.kingdom_card_names)
        self._deck_box.create_if_absent(kingdom_deck)

        output_row = pa.Table.from_pydict(
            {
                "kingdom_uuid": [str(kingdom_deck.nocab_uuid)],
                "ending_type": [self._ending_type(header)],
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
            PosixPath('data/metrics/isotropic/kingdom_ending_type.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _ending_type(self, header: GameHeader) -> str:
        """Classify one game's ending type from its exhausted pile(s).

        Private helper - single consumer is accumulate(). Builds on
        row_utils.is_single_pile_ending() for the single-vs-multi
        branch (Dominion's own end-game rule means a single-pile
        ending can only ever be Province or Colony - see module
        docstring's MULTI_PILE section - so this only needs to
        discriminate those two names, never re-derive the pile count
        itself). Compares by UUID, not by string, since
        header.exhausted_pile_names is English-pluralized
        ("Provinces") - see row_utils.pile_card_uuid_for_name().

        Inputs:
            header: one parsed GameHeader.
        Output: _PROVINCE_ENDING if is_single_pile_ending(header) and
            the single pile resolves to the same card as "Province",
            _COLONY_ENDING if it resolves to "Colony", else
            _MULTI_PILE_ENDING (also the fallback for a single pile
            that resolves to neither, or fails to resolve at all -
            Dominion's own rules say a single-pile ending should never
            be anything but Province or Colony, but this never raises
            on an unexpected observation, only falls back).
        Side effects: none.
        Exceptions: none.
        """
        if not is_single_pile_ending(header):
            return _MULTI_PILE_ENDING

        single_pile_uuid = pile_card_uuid_for_name(
            self._card_binder, header.exhausted_pile_names[0]
        )
        if single_pile_uuid is not None and single_pile_uuid == self._province_uuid:
            return _PROVINCE_ENDING
        if single_pile_uuid is not None and single_pile_uuid == self._colony_uuid:
            return _COLONY_ENDING
        return _MULTI_PILE_ENDING

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
                    "KingdomEndingTypeMetric: unresolved kingdom card name "
                    "%r - excluding it from this kingdom",
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
