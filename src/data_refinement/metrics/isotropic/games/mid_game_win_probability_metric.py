"""Streaming metric: BRAINSTORM.md's "New candidates raised during
human review" section, multi-card/multi-group #6 - given one player's
partial deck at turn T plus the kingdom, predict whether that player
eventually wins. `P(this player eventually wins | partial deck at turn
T, kingdom)`.

A REAL-TIME "WIN PROBABILITY CURVE" DATASET, BUILT AT EVERY TURN
CHECKPOINT: unlike mid_game_next_buy_metric.py/
mid_game_next_trashed_card_metric.py (which skip a turn with nothing to
label), EVERY player-turn in an eligible game_log contributes a row
here - "did this player eventually win" is always answerable, at any
checkpoint, since header_parser.py already guarantees a real winner
exists for any game_log that isn't None. A downstream dojo bucketing
these rows by turn number is what turns this into an actual probability
CURVE - this metric's own job is just producing one labeled example per
checkpoint, same "streaming, no accumulation" shape as every other
per-example classification metric in this container.
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
from src.data_refinement.metrics.isotropic.games.game_log_parser import GameLog, Turn
from src.data_refinement.metrics.isotropic.games.partial_deck import (
    partial_deck_card_uuids,
)
from src.data_refinement.metrics.isotropic.games.row_utils import card_uuid_for_name
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class EventualWinProbabilityMetric:
    """(partial_deck_uuid, kingdom_uuid, player_wins), one row per
    player-turn in every eligible game_log.

    Satisfies the Metric[GameLog] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/mid_game_win_probability.parquet"
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
                ("partial_deck_uuid", pa.string()),
                ("kingdom_uuid", pa.string()),
                ("player_wins", pa.bool_()),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, game_log: GameLog) -> None:
        """Write one row per player-turn in game_log.

        Inputs:
            game_log: one parsed GameLog (game_log_parser.py).
        Output: none.
        Side effects: writes the game's kingdom, and every turn's
            partial deck (as of just before that turn), into
            self._deck_box via create_if_absent(); writes one output
            row per turn, labeled by whether turn.player_nick ==
            game_log.header.winner_nick. Emits one logging.error() per
            unmatched kingdom card name (via _kingdom_deck()).
        Exceptions: none expected beyond a malformed game_log.

        Example:
            >>> metric = EventualWinProbabilityMetric(card_binder, deck_box)
            >>> metric.accumulate(game_log)
            >>> metric.finalize()
        """
        kingdom_deck = self._kingdom_deck(game_log.header.kingdom_card_names)
        self._deck_box.create_if_absent(kingdom_deck)

        for turn in game_log.turns:
            partial_deck = self._partial_deck(game_log, turn)
            self._deck_box.create_if_absent(partial_deck)
            player_wins = turn.player_nick == game_log.header.winner_nick

            output_row = pa.Table.from_pydict(
                {
                    "partial_deck_uuid": [str(partial_deck.nocab_uuid)],
                    "kingdom_uuid": [str(kingdom_deck.nocab_uuid)],
                    "player_wins": [player_wins],
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
            PosixPath('data/metrics/isotropic/mid_game_win_probability.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _partial_deck(self, game_log: GameLog, turn: Turn) -> GenericDeck:
        """Build the GenericDeck for one player-turn's partial deck.

        Private helper - single consumer is accumulate(). Identical
        shape to mid_game_next_buy_metric.py's own _partial_deck() -
        kept as this file's own copy, matching this container's
        established no-cross-file-sharing convention for per-metric
        helpers this small.

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
        (e.g. mid_game_next_buy_metric.py's own _kingdom_deck()) - not
        shared across files.

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
