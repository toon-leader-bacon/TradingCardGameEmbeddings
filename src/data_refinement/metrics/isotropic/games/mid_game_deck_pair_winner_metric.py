"""Streaming metric: BRAINSTORM.md's "New candidates raised during
human review" section, multi-card/multi-group #7 - the mid-game
counterpart of ../summary/deck_pair_winner_metric.py's DeckPairWinnerMetric
(this file's own module docstring) - given both players' partial decks
at the SAME turn checkpoint T, predict which one eventually wins.

RESTRICTED TO REAL 2-PLAYER GAMES, LIKE ITS FLAVOR A COUNTERPART:
game_log_parser.parse_game_log() never itself filters by player count -
this metric does, the same way ../summary/deck_pair_winner_metric.py's
own module docstring documents for its own 2-player restriction (a
"pair" is undefined for 1, 3, or 4 players).

CANONICAL (deck_uuid-SORTED) ORDERING TO AVOID LABEL LEAKAGE - see
../summary/deck_pair_winner_metric.py's own module docstring for why:
the same concern applies here (isotropic's players[] order might carry
a hidden regularity, e.g. automatch host ordering) and the same fix
applies (_sorted_deck_uuids()-style canonical ordering, not raw
players[] order).

TURN CHECKPOINT ALIGNMENT: both players' OWN turn-number counters
(game_log_parser.py's own module docstring - each player's "turn N" is
their own personal count, not a shared round index) are used directly
as the checkpoint key - checkpoint T pairs playerA's "before turn T"
partial deck with playerB's "before turn T" partial deck. Only
checkpoints where BOTH players have actually reached turn T (i.e.
T <= each player's own header_parser.GameHeaderPlayer.turns) contribute
a row - a real 2-player game with an uneven number of turns (the last
player to move in a game that ends via pile exhaustion may have taken
one fewer or more turn than their opponent) simply stops contributing
new checkpoints past whichever player's turns run out first, not
padded or extrapolated.
"""

from pathlib import Path
from typing import ClassVar

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.isotropic.games.game_log_parser import GameLog
from src.data_refinement.metrics.isotropic.games.partial_deck import (
    partial_deck_card_uuids,
)
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_TWO_PLAYERS = 2


class MidGameDeckPairWinnerMetric:
    """(deck_uuid_lo, deck_uuid_hi, lo_wins), one row per turn
    checkpoint reached by both players of a real 2-player game.

    Satisfies the Metric[GameLog] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/mid_game_deck_pair_winner.parquet"
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
            deck_box: the metrics-private DeckBox every partial deck
                this metric sees is written into.
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
                    ("deck_uuid_lo", pa.string()),
                    ("deck_uuid_hi", pa.string()),
                    ("lo_wins", pa.bool_()),
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
        """Write one row per turn checkpoint reached by both players,
        for a real 2-player game_log.

        Inputs:
            game_log: one parsed GameLog (game_log_parser.py).
        Output: none.
        Side effects: skips entirely (writes nothing) if
            len(game_log.header.players) != 2 (module docstring). For
            every checkpoint T both players reached, writes both
            partial decks into self._deck_box via create_if_absent(),
            then buffers one output row, canonically ordered by
            deck_uuid (module docstring's CANONICAL ORDERING note).
        Exceptions: none expected beyond a malformed game_log.

        Example:
            >>> metric = MidGameDeckPairWinnerMetric(card_binder, deck_box)
            >>> metric.accumulate(game_log)
            >>> metric.finalize()
        """
        players = game_log.header.players
        if len(players) != _TWO_PLAYERS:
            return

        player_a, player_b = players
        winner_nick = game_log.header.winner_nick
        max_checkpoint = min(player_a.turns, player_b.turns)

        for checkpoint in range(1, max_checkpoint + 1):
            deck_a, deck_b = self._checkpoint_decks(
                game_log, player_a.nick, player_b.nick, checkpoint
            )
            # Identical decks (e.g. both players' untouched starting deck
            # at checkpoint 1) make "which deck won" undefined - and would
            # leak players[] order through the lo/hi tie-break.
            if deck_a.nocab_uuid == deck_b.nocab_uuid:
                continue
            self._deck_box.create_if_absent(deck_a)
            self._deck_box.create_if_absent(deck_b)

            deck_lo, deck_hi, lo_wins = self._canonical_pair(
                deck_a, deck_b, player_a.nick, player_b.nick, winner_nick
            )
            output_row = {
                "deck_uuid_lo": str(deck_lo.nocab_uuid),
                "deck_uuid_hi": str(deck_hi.nocab_uuid),
                "lo_wins": lo_wins,
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
            PosixPath('data/metrics/isotropic/mid_game_deck_pair_winner.parquet')
        """
        self._writer.close()
        return self._output_path

    def _checkpoint_decks(
        self,
        game_log: GameLog,
        nick_a: str,
        nick_b: str,
        checkpoint: int,
    ) -> tuple[GenericDeck, GenericDeck]:
        """Build both players' partial-deck GenericDecks for one turn
        checkpoint.

        Private helper - single consumer is accumulate().

        Inputs:
            game_log: the enclosing GameLog.
            nick_a: the first player's nick.
            nick_b: the second player's nick.
            checkpoint: the shared "before this turn number" checkpoint
                (module docstring's TURN CHECKPOINT ALIGNMENT note).
        Output: (deck_a, deck_b), each built over
            partial_deck.partial_deck_card_uuids() for its own nick at
            this checkpoint. source_game=GameId.DOMINION, provenance
            left at its default (None) for both.
        Side effects: whatever partial_deck_card_uuids() does for each
            nick (logs unmatched names, never raises).
        Exceptions: none.
        """
        decks = []
        for nick in (nick_a, nick_b):
            card_nocab_uuids = partial_deck_card_uuids(
                self._card_binder, game_log, nick, checkpoint
            )
            decks.append(
                GenericDeck(
                    nocab_uuid=deck_uuid_from_cards(card_nocab_uuids),
                    source_game=GameId.DOMINION,
                    name="isotropic partial deck",
                    card_nocab_uuids=card_nocab_uuids,
                )
            )
        return decks[0], decks[1]

    def _canonical_pair(
        self,
        deck_a: GenericDeck,
        deck_b: GenericDeck,
        nick_a: str,
        nick_b: str,
        winner_nick: str,
    ) -> tuple[GenericDeck, GenericDeck, bool]:
        """Order (deck_a, deck_b) into (deck_lo, deck_hi) by nocab_uuid,
        and compute whether deck_lo's own owner is the eventual winner.

        Private helper - single consumer is accumulate(). Module
        docstring's CANONICAL ORDERING note - the ordering must never be
        based on nick_a/nick_b's own positional order (isotropic's own
        players[] order), only on the decks' own uuids.

        Inputs:
            deck_a, deck_b: this checkpoint's two partial decks.
            nick_a, nick_b: the respective owning players' nicks.
            winner_nick: game_log.header.winner_nick.
        Output: (deck_lo, deck_hi, lo_wins) - deck_lo is whichever of
            deck_a/deck_b has the lexicographically smaller
            str(nocab_uuid); lo_wins is True iff deck_lo's own owning
            nick equals winner_nick.
        Side effects: none.
        Exceptions: none.
        """
        if str(deck_a.nocab_uuid) <= str(deck_b.nocab_uuid):
            return deck_a, deck_b, nick_a == winner_nick
        return deck_b, deck_a, nick_b == winner_nick
