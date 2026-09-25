"""Streaming metric: BRAINSTORM.md's "Multi Group Metrics" shortlist -
given both decks from a 2-player game, predict which one won.

CANONICAL ORDERING TO AVOID LABEL LEAKAGE: the two decks are written in
a FIXED order (sorted by deck_uuid, not by isotropic's own players[]
order) - if players[] order carried any hidden regularity (e.g.
whoever isotropic lists first tends to be the automatch "host"), a
model trained on always-players[0]-vs-players[1] pairs could pick that
up instead of real deck signal. _sorted_deck_uuids() below is where
this ordering is fixed; whoever implements it must not "simplify" this
back to raw players[] order.

Restricted to real 2-player, both-eligible games - BRAINSTORM.md's
known-biases note on solo/practice games and resignations both apply:
a 1-player game has no opponent to pair against, and a resigned
player's end.deck doesn't exist to build a deck from.
"""

from pathlib import Path
from typing import ClassVar

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    deck_for_player,
    eligible_player_entries,
)
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_TWO_PLAYERS = 2


class DeckPairWinnerMetric:
    """(deck_uuid_lo, deck_uuid_hi, lo_won), one row per real 2-player
    game where both players' final decks are resolvable.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/deck_pair_winner.parquet"
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
            deck_box: the metrics-private DeckBox both decks in each
                pair are written into.
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
                    ("lo_won", pa.bool_()),
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
        """Write one (deck_uuid_lo, deck_uuid_hi, lo_won) row for one
        real 2-player game with both decks resolvable.

        Inputs:
            row: one parsed Flavor A summary row, carrying "players".
        Output: none.
        Side effects: buffers exactly one row into the open
            ParquetBuilder when exactly _TWO_PLAYERS eligible players
            exist for this row (row_utils.eligible_player_entries()) -
            writes nothing for a solo game, a 3-4p game (see
            multiplayer_placement_metric.py for that shape instead), or
            a 2-player game where either player resigned. Writes both
            decks into self._deck_box via create_if_absent().
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = DeckPairWinnerMetric(card_binder, deck_box)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        players = eligible_player_entries(row)
        if len(players) != _TWO_PLAYERS:
            return

        deck_a = self._deck_for_player(players[0])
        deck_b = self._deck_for_player(players[1])
        self._deck_box.create_if_absent(deck_a)
        self._deck_box.create_if_absent(deck_b)

        lo_deck, hi_deck = self._sorted_deck_uuids(deck_a, deck_b)
        lo_player = (
            players[0] if lo_deck.nocab_uuid == deck_a.nocab_uuid else players[1]
        )

        self._writer.write_row(
            {
                "deck_uuid_lo": str(lo_deck.nocab_uuid),
                "deck_uuid_hi": str(hi_deck.nocab_uuid),
                "lo_won": lo_player["rank"] == 1,
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
            PosixPath('data/metrics/isotropic/deck_pair_winner.parquet')
        """
        self._writer.close()
        return self._output_path

    def _sorted_deck_uuids(
        self, deck_a: GenericDeck, deck_b: GenericDeck
    ) -> tuple[GenericDeck, GenericDeck]:
        """Order two decks canonically by deck_uuid, so output row
        ordering never leaks isotropic's own players[] order.

        Private helper - single consumer is accumulate(). See module
        docstring's CANONICAL ORDERING note - this is the one place
        that ordering is enforced; do not bypass it by writing
        players[] order directly in accumulate().

        Inputs:
            deck_a: one player's deck (either order).
            deck_b: the other player's deck (either order).
        Output: (lo, hi) - deck_a and deck_b reordered so
            str(lo.nocab_uuid) < str(hi.nocab_uuid).
        Side effects: none.
        Exceptions: none.
        """
        if str(deck_a.nocab_uuid) <= str(deck_b.nocab_uuid):
            return deck_a, deck_b
        return deck_b, deck_a

    def _deck_for_player(self, player_entry: dict) -> GenericDeck:
        """Build the GenericDeck for one eligible player's end.deck.

        Private helper - single consumer is accumulate(). Delegates to
        row_utils.deck_for_player() - the shared per-player deck
        builder now centralized there, since this file,
        full_deck_win_prediction_metric.py, and
        multiplayer_placement_metric.py all reached this identical
        logic independently (the "third consumer" trigger this
        method's own skeleton docstring flagged).

        Inputs:
            player_entry: one players[] entry with a real "end" block.
        Output: a GenericDeck, source_game=GameId.DOMINION. provenance
            is left as its default (None) - this is a private,
            metrics-only DeckBox entry, not a canonical ingested deck,
            mirroring ../../sts_gg/deck_label_metric.py's own GenericDeck
            construction (which likewise omits provenance entirely).
        Side effects: none (the caller writes it into deck_box).
        Exceptions: none.
        """
        return deck_for_player(self._card_binder, player_entry)
