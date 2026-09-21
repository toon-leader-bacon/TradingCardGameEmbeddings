"""Streaming metric: BRAINSTORM.md's "More Flavor B-only candidates"
section, multi-card #26 - given the kingdom, predict the eventual
winner's opening buy (1-2 cards, or 1 card when the second slot was
"nothing").

VARIABLE-SET LABEL: like ../summary/kingdom_veto_prediction_metric.py,
LABEL_VALUES is not fixed here - the opening buy(s) range over
dominiontabs' whole card pool (kingdom cards plus Silver - a
non-kingdom card that's still a legal opening buy), so the label
column is a list of 1-2 card uuids, not a single fixed-vocabulary
string. "nothing" contributes no entry to that list (an empty second
slot is absence, not a sentinel value).

Restricted to natural kingdoms - see header_parser.py's
GameHeader.is_natural_kingdom.
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
    winner_player,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class KingdomOpeningBuyPredictionMetric:
    """(kingdom_uuid, opening_card_uuids), one row per natural-kingdom
    game, written as soon as accumulate() sees it.

    Satisfies the Metric[GameHeader] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/kingdom_opening_buy_prediction.parquet"
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
                ("opening_card_uuids", pa.list_(pa.string())),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, header: GameHeader) -> None:
        """Write one (kingdom_uuid, opening_card_uuids) row for one
        natural-kingdom game.

        Inputs:
            header: one parsed GameHeader (header_parser.py).
        Output: none.
        Side effects: writes exactly one row to the open ParquetWriter
            when header.is_natural_kingdom is True and at least one of
            the winner's opening buy names resolves - writes nothing
            for a generator-constrained kingdom, or if neither opening
            buy name resolves (a real "nothing to predict" outcome).
            Writes the kingdom into self._deck_box via
            create_if_absent(). Emits one logging.error() per opening
            buy name or kingdom card name that fails to resolve (see
            _kingdom_deck()).
        Exceptions: none expected beyond a malformed header.

        Example:
            >>> metric = KingdomOpeningBuyPredictionMetric(card_binder, deck_box)
            >>> metric.accumulate(header)
            >>> metric.finalize()
        """
        if not header.is_natural_kingdom:
            return

        winner = winner_player(header)
        opening_card_uuids = [
            card_uuid
            for card_uuid in (
                self._card_uuid_or_log(name)
                for name in winner.opening_buy_names
                if name is not None
            )
            if card_uuid is not None
        ]
        if not opening_card_uuids:
            return

        kingdom_deck = self._kingdom_deck(header.kingdom_card_names)
        self._deck_box.create_if_absent(kingdom_deck)

        output_row = pa.Table.from_pydict(
            {
                "kingdom_uuid": [str(kingdom_deck.nocab_uuid)],
                "opening_card_uuids": [[str(u) for u in opening_card_uuids]],
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
            PosixPath('data/metrics/isotropic/kingdom_opening_buy_prediction.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _card_uuid_or_log(self, card_name: str) -> UUID | None:
        """Resolve one opening-buy card name, logging on failure.

        Private helper - single consumer is accumulate().

        Inputs:
            card_name: one non-None opening_buy_names entry.
        Output: the matching nocab_uuid, or None if unresolved.
        Side effects: emits one logging.error() call when unresolved.
        Exceptions: none.
        """
        card_uuid = card_uuid_for_name(self._card_binder, card_name)
        if card_uuid is None:
            _logger.error(
                "KingdomOpeningBuyPredictionMetric: unresolved opening buy "
                "name %r - excluding it from this row's label",
                card_name,
            )
        return card_uuid

    def _kingdom_deck(self, kingdom_names: tuple[str, ...]) -> GenericDeck:
        """Build the GenericDeck for one game's kingdom card group.

        Private helper - single consumer is accumulate(). Same shape
        as every other "kingdom as a card group" builder in this
        project (e.g. ../summary/kingdom_game_length_metric.py's own
        _kingdom_deck()) - not shared across the two subpackages, same
        reasoning as row_utils.py's own module docstring gives for
        card_uuid_for_name().

        Inputs:
            kingdom_names: this game's dealt kingdom_card_names.
        Output: a GenericDeck over the resolved kingdom names -
            unresolved names are skipped, not raised on (mirroring
            card_resolution.card_uuid_for_name()'s per-name contract),
            so this method's own docstring does NOT promise
            len(card_nocab_uuids) == len(kingdom_names).
            source_game=GameId.DOMINION, provenance left at its
            default (None) - a private, metrics-only DeckBox entry.
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
                    "KingdomOpeningBuyPredictionMetric: unresolved kingdom "
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
