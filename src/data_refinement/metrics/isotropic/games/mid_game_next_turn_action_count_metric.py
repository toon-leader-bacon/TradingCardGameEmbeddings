"""Streaming metric: BRAINSTORM.md's "New candidates raised during
human review" section, multi-card/multi-group #5 - given a player's
partial deck at turn T, predict how many Action cards they play on
their OWN next turn. `E[number of Action cards played next turn |
partial deck at turn T]` - an engine-momentum signal.

ACTION-TYPE FILTERING HAPPENS HERE, NOT IN game_log_parser.py:
game_log_parser.Turn.cards_played deliberately records every card
played of ANY type (module docstring there) - this metric is the
reason why: it needs a real CardBinder lookup (dominiontabs' own
"Action" in card.raw_content["types"], the same list-valued field
../../dominiontabs/type_mask_metric.py already reads) to decide which
plays count, which the parser itself has no access to.

The predictor is the partial deck as of just BEFORE turn T (the deck
the player started turn T with), so turn T's own gains are not in it;
the label is the Action plays on the same player's turn T+1.

ONE ROW PER PLAYER-TURN THAT HAS A NEXT TURN OF ITS OWN: the player's
LAST turn of the game (or their last turn before the game_log
otherwise ends) contributes no row - there is no "next turn" to count
actions in.
"""

from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.isotropic.games.game_log_parser import GameLog, Turn
from src.data_refinement.metrics.isotropic.games.partial_deck import (
    card_uuid_for_quantity_name,
    partial_deck_card_uuids,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_ACTION_TYPE_NAME = "Action"


class NextTurnActionCountMetric:
    """(partial_deck_uuid, next_turn_action_count), one row per
    player-turn that has a following turn of the same player's own.

    Satisfies the Metric[GameLog] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/mid_game_next_turn_action_count.parquet"
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
                against, and to check each played card's own "Action"
                membership in raw_content["types"] - must already have
                dominiontabs' cards ingested (this class never writes
                to it).
            deck_box: the metrics-private DeckBox each turn's partial
                deck is written into.
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
                ("next_turn_action_count", pa.int64()),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, game_log: GameLog) -> None:
        """Write one row per player-turn that has a following turn of
        the same player's own.

        Inputs:
            game_log: one parsed GameLog (game_log_parser.py).
        Output: none.
        Side effects: for every turn across every player whose own next
            turn (turn_number + 1) is also present in game_log.turns,
            writes the partial deck (as of just before this turn) into
            self._deck_box via create_if_absent(), then writes one
            output row whose label is the count of Action-typed cards
            played on that next turn. Emits one logging.error() per
            unmatched played-card name (see _action_play_count()).
        Exceptions: none expected beyond a malformed game_log.

        Example:
            >>> metric = NextTurnActionCountMetric(card_binder, deck_box)
            >>> metric.accumulate(game_log)
            >>> metric.finalize()
        """
        turns_by_player: dict[str, dict[int, Turn]] = {}
        for turn in game_log.turns:
            turns_by_player.setdefault(turn.player_nick, {})[turn.turn_number] = turn

        for turn in game_log.turns:
            next_turn = turns_by_player[turn.player_nick].get(turn.turn_number + 1)
            if next_turn is None:
                continue

            partial_deck = self._partial_deck(game_log, turn)
            self._deck_box.create_if_absent(partial_deck)
            action_count = self._action_play_count(next_turn)

            output_row = pa.Table.from_pydict(
                {
                    "partial_deck_uuid": [str(partial_deck.nocab_uuid)],
                    "next_turn_action_count": [action_count],
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
            PosixPath('data/metrics/isotropic/mid_game_next_turn_action_count.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _action_play_count(self, turn: Turn) -> int:
        """Count how many of turn.cards_played match a card whose
        dominiontabs raw_content["types"] includes "Action".

        Private helper - single consumer is accumulate().

        Inputs:
            turn: the NEXT turn (module docstring) whose plays are
                being counted.
        Output: the count - a CardQuantity with count=N that matches
            to an Action-typed card contributes N, not 1. A quantity
            that has no match contributes 0, not an exception.
        Side effects: emits one logging.error() call per unmatched
            played-card name.
        Exceptions: none.
        """
        action_play_count = 0
        for quantity in turn.cards_played:
            card_uuid = card_uuid_for_quantity_name(self._card_binder, quantity)
            if card_uuid is None:
                continue
            card = self._card_binder.get_by_uuid(card_uuid)
            if card is not None and _ACTION_TYPE_NAME in card.raw_content["types"]:
                action_play_count += quantity.count
        return action_play_count

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
