"""Streaming metric: BRAINSTORM.md's "New candidates raised during
human review" section, multi-card/multi-group #4 - given a player's
partial deck at turn T, predict which card they trash on turn T,
CONDITIONED ON A TRASH HAPPENING AT ALL. `P(card_trashed = X | deck at
turn T, a trash event occurs on this turn)`.

NOT predicting whether a trash occurs - only turns with at least one
matchable game_log_parser.Turn.cards_trashed entry contribute a row,
same "skip, don't label a negative" shape as
../pile_exhaustion_rate_metric.py's own natural-kingdom-only filter (a
different kind of conditioning, same idea: this metric only ever
answers "given a trash happened, which card").

NO KINGDOM INPUT, UNLIKE mid_game_next_buy_metric.py: BRAINSTORM.md's
own framing for this metric names only "partial deck at turn T," not
the kingdom - since what CAN be trashed is fully determined by what's
already in the partial deck, the kingdom carries no additional
constraint the way it does for a buy (which ranges over the whole
kingdom-plus-basics pool, not just what's already owned).
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
    distinct_card_uuids,
    partial_deck_card_uuids,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId


class NextTrashedCardMetric:
    """(partial_deck_uuid, next_trashed_card_uuids), one row per
    player-turn with at least one matchable trash.

    Satisfies the Metric[GameLog] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/mid_game_next_trashed_card.parquet"
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
                ("next_trashed_card_uuids", pa.list_(pa.string())),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, game_log: GameLog) -> None:
        """Write one row per player-turn in game_log with at least one
        matchable trash.

        Inputs:
            game_log: one parsed GameLog (game_log_parser.py).
        Output: none.
        Side effects: for every turn across every player with a
            non-empty, at-least-partially-matchable cards_trashed,
            writes the partial deck (as of just before that turn) into
            self._deck_box via create_if_absent(), then writes one
            output row. Emits one logging.error() per unmatched
            trashed-card name (via partial_deck.py).
        Exceptions: none expected beyond a malformed game_log.

        Example:
            >>> metric = NextTrashedCardMetric(card_binder, deck_box)
            >>> metric.accumulate(game_log)
            >>> metric.finalize()
        """
        for turn in game_log.turns:
            next_trashed_card_uuids = distinct_card_uuids(
                self._card_binder, turn.cards_trashed
            )
            if not next_trashed_card_uuids:
                continue

            partial_deck = self._partial_deck(game_log, turn)
            self._deck_box.create_if_absent(partial_deck)

            output_row = pa.Table.from_pydict(
                {
                    "partial_deck_uuid": [str(partial_deck.nocab_uuid)],
                    "next_trashed_card_uuids": [
                        [str(u) for u in next_trashed_card_uuids]
                    ],
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
            PosixPath('data/metrics/isotropic/mid_game_next_trashed_card.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _partial_deck(self, game_log: GameLog, turn: Turn) -> GenericDeck:
        """Build the GenericDeck for one player-turn's partial deck.

        Private helper - single consumer is accumulate(). Identical
        shape to mid_game_next_buy_metric.py's own _partial_deck() -
        kept as this file's own copy, not shared, same reasoning
        row_utils.py's own module docstring gives for
        card_uuid_for_name() vs. _kingdom_deck()-style per-file helpers
        (contrast partial_deck.distinct_card_uuids(), promoted to a
        shared function instead, since it has zero per-metric
        variation - see that function's own module docstring).

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
