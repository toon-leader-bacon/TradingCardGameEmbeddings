"""Streaming metric: BRAINSTORM.md's "Multi Group Metrics" shortlist -
given every deck in a 3-4 player game, predict the full finishing
order.

WHOLE-GAME ELIGIBILITY, NOT PER-PLAYER: unlike every other metric in
this container, a single resignation anywhere in a 3-4p game discards
the WHOLE game, not just that one player - BRAINSTORM.md multi-card #9
flags "a real design question on how ties/resignations mid-ranking
should be handled," and this skeleton resolves it to the simplest
option (require every player to have reached a real end) rather than
guessing at a partial-ranking scheme; a future revision could relax
this once someone has a concrete need for partial multiplayer rankings.

Ranking label as parallel arrays, canonically ordered by deck_uuid (not
by isotropic's own players[] order or by rank) for the same
leakage-avoidance reason deck_pair_winner_metric.py sorts its pair -
see _sorted_by_deck_uuid()'s docstring.
"""

from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.summary.row_utils import deck_for_player
from src.schema.card import GenericDeck

_MIN_MULTIPLAYER_COUNT = 3
_MAX_MULTIPLAYER_COUNT = 4


class MultiplayerPlacementMetric:
    """(deck_uuids, ranks), one row per 3-4p game where every player
    reached a real end - both lists parallel and canonically ordered
    by deck_uuid.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/multiplayer_placement.parquet"
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
            deck_box: the metrics-private DeckBox every deck this
                metric sees is written into.
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
                ("deck_uuids", pa.list_(pa.string())),
                ("ranks", pa.list_(pa.int64())),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Write one (deck_uuids, ranks) row for one 3-4p game where
        every player reached a real end.

        Inputs:
            row: one parsed Flavor A summary row, carrying "players".
        Output: none.
        Side effects: writes exactly one row to the open ParquetWriter
            when _MIN_MULTIPLAYER_COUNT <= len(row["players"]) <=
            _MAX_MULTIPLAYER_COUNT AND every one of those players has a
            real "end" block - writes nothing for a 1-2p game or a
            multiplayer game with any resignation (module docstring's
            WHOLE-GAME ELIGIBILITY note). Writes every player's deck
            into self._deck_box via create_if_absent().
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = MultiplayerPlacementMetric(card_binder, deck_box)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        players = row["players"]
        if not (_MIN_MULTIPLAYER_COUNT <= len(players) <= _MAX_MULTIPLAYER_COUNT):
            return
        if any("end" not in player for player in players):
            # Module docstring's WHOLE-GAME ELIGIBILITY note - one
            # resignation anywhere discards the whole game.
            return

        decks = [self._deck_for_player(player) for player in players]
        ranks = [player["rank"] for player in players]
        for deck in decks:
            self._deck_box.create_if_absent(deck)

        deck_uuid_strs, sorted_ranks = self._sorted_by_deck_uuid(decks, ranks)

        output_row = pa.Table.from_pydict(
            {
                "deck_uuids": [deck_uuid_strs],
                "ranks": [sorted_ranks],
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
            PosixPath('data/metrics/isotropic/multiplayer_placement.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _sorted_by_deck_uuid(
        self, decks: list[GenericDeck], ranks: list[int]
    ) -> tuple[list[str], list[int]]:
        """Order this game's decks and their ranks together by
        deck_uuid, so output row ordering never leaks isotropic's own
        players[] order (mirrors deck_pair_winner_metric.py's
        _sorted_deck_uuids() for the 2-player case).

        Private helper - single consumer is accumulate().

        Inputs:
            decks: this game's resolved GenericDeck per player, in
                row["players"] order.
            ranks: the matching rank per entry of decks, same order.
        Output: (deck_uuid strings, ranks) - both reordered together so
            deck_uuid strings are in ascending sorted order.
        Side effects: none.
        Exceptions: none.
        """
        pairs = sorted(zip(decks, ranks), key=lambda pair: str(pair[0].nocab_uuid))
        deck_uuid_strs = [str(deck.nocab_uuid) for deck, _ in pairs]
        sorted_ranks = [rank for _, rank in pairs]
        return deck_uuid_strs, sorted_ranks

    def _deck_for_player(self, player_entry: dict) -> GenericDeck:
        """Build the GenericDeck for one player's end.deck.

        Private helper - single consumer is accumulate(). Delegates to
        row_utils.deck_for_player() - the shared per-player deck
        builder now centralized there, since this file,
        full_deck_win_prediction_metric.py, and
        deck_pair_winner_metric.py all reached this identical logic
        independently (the "third consumer" trigger this method's own
        skeleton docstring flagged).

        Inputs:
            player_entry: one players[] entry with a real "end" block.
        Output: a GenericDeck - see row_utils.deck_for_player()'s own
            docstring for the full contract.
        Side effects: none (the caller writes it into deck_box).
        Exceptions: none.
        """
        return deck_for_player(self._card_binder, player_entry)
