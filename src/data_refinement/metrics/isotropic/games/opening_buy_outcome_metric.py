"""Streaming metric: BRAINSTORM.md's "More Flavor B-only candidates"
section, multi-card #27 - given only a player's own opening buy (plus
the kingdom), predict whether that player won.

PER PLAYER, NOT WINNER-ONLY: the inverse question
kingdom_opening_buy_prediction_metric.py asks (kingdom -> the winner's
opening) only ever has one positive example per game; this metric asks
the opposite direction (opening -> outcome) for EVERY player in the
game, since every player's own opening/outcome pair is a real training
example regardless of whether they won - matching BRAINSTORM.md's own
framing ("how much of the eventual outcome is already determined" by
an opening, which needs both winning and losing openings to answer).

OPENING BUY AS A DECKBOX GROUP: mirrors
../summary/kingdom_veto_prediction_metric.py's candidate-pool
convention - a player's 1-2 opening buys aren't a deck (never scored
on their own), so they're stored via deck_box as a generic "named card
group," content-addressed the same way a real deck is.

Restricted to natural kingdoms - see header_parser.py's
GameHeader.is_natural_kingdom.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.isotropic.games.header_parser import (
    GameHeader,
    GameHeaderPlayer,
)
from src.data_refinement.metrics.isotropic.games.row_utils import card_uuid_for_name
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class OpeningBuyOutcomeMetric:
    """(opening_group_uuid, kingdom_uuid, won), one row per player per
    natural-kingdom game with at least one resolvable opening buy.

    Satisfies the Metric[GameHeader] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/opening_buy_outcome.parquet"
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
            deck_box: the metrics-private DeckBox every opening-buy
                group this metric sees is written into.
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
                    ("opening_group_uuid", pa.string()),
                    ("kingdom_uuid", pa.string()),
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

    def accumulate(self, header: GameHeader) -> None:
        """Write one (opening_group_uuid, kingdom_uuid, won) row per
        player in one natural-kingdom game whose opening has at least
        one resolvable card.

        Inputs:
            header: one parsed GameHeader (header_parser.py).
        Output: none.
        Side effects: buffers one row into the open ParquetBuilder per
            player with a resolvable opening, when header is a natural
            kingdom - writes nothing for a generator-constrained
            kingdom. Writes each player's opening group and the
            kingdom into self._deck_box via create_if_absent(). Emits
            one logging.error() per opening buy name or kingdom card
            name that fails to resolve (see _opening_group_deck()/
            _kingdom_deck()).
        Exceptions: none expected beyond a malformed header.

        Example:
            >>> metric = OpeningBuyOutcomeMetric(card_binder, deck_box)
            >>> metric.accumulate(header)
            >>> metric.finalize()
        """
        if not header.is_natural_kingdom:
            return

        kingdom_deck = self._kingdom_deck(header.kingdom_card_names)
        kingdom_written = False

        for player in header.players:
            opening_deck = self._opening_group_deck(player)
            if opening_deck is None:
                continue

            if not kingdom_written:
                self._deck_box.create_if_absent(kingdom_deck)
                kingdom_written = True
            self._deck_box.create_if_absent(opening_deck)

            output_row = {
                "opening_group_uuid": str(opening_deck.nocab_uuid),
                "kingdom_uuid": str(kingdom_deck.nocab_uuid),
                "won": player.nick == header.winner_nick,
            }
            self._writer.write_row(output_row)

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
            PosixPath('data/metrics/isotropic/opening_buy_outcome.parquet')
        """
        self._writer.close()
        return self._output_path

    def _opening_group_deck(self, player: GameHeaderPlayer) -> GenericDeck | None:
        """Build the GenericDeck for one player's opening buy(s).

        Private helper - single consumer is accumulate().

        Inputs:
            player: one GameHeaderPlayer.
        Output: a GenericDeck over player.opening_buy_names' resolved,
            non-None entries, or None if neither entry resolves (a
            real "nothing to record" outcome - e.g. both names failed
            to resolve, or the player's own second slot was "nothing"
            and the first also failed). source_game=GameId.DOMINION,
            provenance left at its default (None).
        Side effects: none (the caller writes it into deck_box). Emits
            one logging.error() per opening buy name that fails to
            resolve.
        Exceptions: none.
        """
        card_nocab_uuids: list[UUID] = []
        for card_name in player.opening_buy_names:
            if card_name is None:
                continue
            card_uuid = card_uuid_for_name(self._card_binder, card_name)
            if card_uuid is None:
                _logger.error(
                    "OpeningBuyOutcomeMetric: unresolved opening buy name "
                    "%r - excluding it from %s's opening group",
                    card_name,
                    player.nick,
                )
                continue
            card_nocab_uuids.append(card_uuid)

        if not card_nocab_uuids:
            return None

        deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
        return GenericDeck(
            nocab_uuid=deck_uuid,
            source_game=GameId.DOMINION,
            name=f"isotropic {player.nick} opening buy",
            card_nocab_uuids=card_nocab_uuids,
        )

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
                    "OpeningBuyOutcomeMetric: unresolved kingdom card name "
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
