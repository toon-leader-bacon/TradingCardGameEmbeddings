"""Streaming metric: BRAINSTORM.md's "New candidates raised during
human review" section, multi-card/multi-group #3 - given a player's
partial deck at turn T plus the kingdom, predict what they buy on turn
T. `P(next card bought = X | partial deck at turn T, kingdom)`.

ONE ROW PER PLAYER-TURN WITH A REAL BUY: a turn with zero buys (e.g. a
dead early turn, or the game's very first "opening" turn before this
pass's own definition of "partial deck" has anything interesting in it)
contributes no row - nothing to predict. Every eligible player
contributes multiple rows across the game (one per turn they bought
something), unlike header-level metrics that contribute at most one row
per player per game.

VARIABLE-SET LABEL, LIKE kingdom_opening_buy_prediction_metric.py: a
turn's buys can name more than one distinct card (module docstring's
"never assumed exactly one" note on game_log_parser.Turn.cards_bought),
so next_buy_card_uuids is a list, not a single uuid - deduplicated by
card identity (buying 3 Coppers in one turn is one label entry, not
three), since this is "which card(s)," not "how many."
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.isotropic.games.game_log_parser import GameLog, Turn
from src.data_refinement.metrics.isotropic.games.partial_deck import (
    distinct_card_uuids,
    partial_deck_card_uuids,
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


class NextBuyPredictionMetric:
    """(partial_deck_uuid, kingdom_uuid, next_buy_card_uuids), one row
    per player-turn with at least one matchable buy.

    Satisfies the Metric[GameLog] Protocol (../../metric.py)
    structurally - a NEW raw-row type for this container (every other
    games/ metric today is Metric[GameHeader]).
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/mid_game_next_buy.parquet"
    )

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to look up isotropic card names
                against - must already have dominiontabs' cards
                ingested (this class never writes to it).
            deck_box: the metrics-private DeckBox both each turn's
                partial deck and the game's kingdom are written into.
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
                    ("partial_deck_uuid", pa.string()),
                    ("kingdom_uuid", pa.string()),
                    ("next_buy_card_uuids", pa.list_(pa.string())),
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

    def accumulate(self, game_log: GameLog) -> None:
        """Write one row per player-turn in game_log with at least one
        matchable buy.

        Inputs:
            game_log: one parsed GameLog (game_log_parser.py).
        Output: none.
        Side effects: for every turn across every player with a
            non-empty, at-least-partially-matchable cards_bought,
            writes the partial deck (as of just before that turn) and
            the game's kingdom into self._deck_box via
            create_if_absent(), then buffers one output row. Emits one
            logging.error() per unmatched buy or kingdom card name (see
            _kingdom_deck() and partial_deck.py).
        Exceptions: none expected beyond a malformed game_log.

        Example:
            >>> metric = NextBuyPredictionMetric(card_binder, deck_box)
            >>> metric.accumulate(game_log)
            >>> metric.finalize()
        """
        kingdom_deck = self._kingdom_deck(game_log.header.kingdom_card_names)

        for turn in game_log.turns:
            next_buy_card_uuids = distinct_card_uuids(
                self._card_binder, turn.cards_bought
            )
            if not next_buy_card_uuids:
                continue

            self._deck_box.create_if_absent(kingdom_deck)
            partial_deck = self._partial_deck(game_log, turn)
            self._deck_box.create_if_absent(partial_deck)

            output_row = {
                "partial_deck_uuid": str(partial_deck.nocab_uuid),
                "kingdom_uuid": str(kingdom_deck.nocab_uuid),
                "next_buy_card_uuids": [str(u) for u in next_buy_card_uuids],
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
            PosixPath('data/metrics/isotropic/mid_game_next_buy.parquet')
        """
        self._writer.close()
        return self._output_path

    def _partial_deck(self, game_log: GameLog, turn: Turn) -> GenericDeck:
        """Build the GenericDeck for one player-turn's partial deck.

        Private helper - single consumer is accumulate(). Same
        "content-addressed, provenance=None" convention as every other
        GenericDeck this container mints - two different games/turns
        that happen to reach an identical card multiset dedupe to the
        same DeckBox entry, which is intended, not a bug.

        Inputs:
            game_log: the enclosing GameLog.
            turn: the specific turn to compute the "just before" state
                for.
        Output: a GenericDeck over partial_deck.partial_deck_card_uuids()'s
            result for (turn.player_nick, turn.turn_number).
            source_game=GameId.DOMINION, provenance left at its default
            (None).
        Side effects: whatever partial_deck_card_uuids() does (logs
            unmatched names, never raises).
        Exceptions: none.
        """
        card_nocab_uuids = partial_deck_card_uuids(
            self._card_binder, game_log, turn.player_nick, turn.turn_number
        )
        return GenericDeck(
            nocab_uuid=deck_uuid_from_cards(card_nocab_uuids),
            source_game=GameId.DOMINION,
            name="isotropic partial deck",
            card_nocab_uuids=card_nocab_uuids,
        )

    def _kingdom_deck(self, kingdom_names: tuple[str, ...]) -> GenericDeck:
        """Build the GenericDeck for one game's kingdom card group.

        Private helper - single consumer is accumulate(). Same shape as
        every other "kingdom as a card group" builder in this container
        (e.g. kingdom_opening_buy_prediction_metric.py's own
        _kingdom_deck()) - not shared across files, same reasoning as
        row_utils.py's own module docstring gives for
        card_uuid_for_name().

        Inputs:
            kingdom_names: this game's dealt kingdom_card_names.
        Output: a GenericDeck over the matched kingdom names -
            unmatched names are skipped, not raised on.
            source_game=GameId.DOMINION, provenance left at its default
            (None).
        Side effects: none. Emits one logging.error() per kingdom card
            name that has no match.
        Exceptions: none.
        """
        card_nocab_uuids: list[UUID] = []
        for card_name in kingdom_names:
            card_uuid = card_uuid_for_name(self._card_binder, card_name)
            if card_uuid is None:
                _logger.error(
                    "%s: unmatched kingdom card name %r - excluding it "
                    "from this kingdom",
                    type(self).__name__,
                    card_name,
                )
                continue
            card_nocab_uuids.append(card_uuid)

        return GenericDeck(
            nocab_uuid=deck_uuid_from_cards(card_nocab_uuids),
            source_game=GameId.DOMINION,
            name="isotropic kingdom",
            card_nocab_uuids=card_nocab_uuids,
        )
