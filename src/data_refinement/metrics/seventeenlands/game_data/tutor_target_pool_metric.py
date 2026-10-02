"""TutorTargetPoolMetric - BRAINSTORM.md's multi-card metric "Predict
Tutor Targets Given Deck": given one game's full draft pool
(deck_<name> union sideboard_<name>), label each pool card with whether
it appears in tutored_<name> that game.

A vectorized Metric[GameDataChunk] (see game_data/README.md).
Per chunk it lines the deck, sideboard and tutored zones up
over one card axis (the distinct cards of deck_ and sideboard_ columns,
via ZoneCounts.present_for()), so the chunk's pool is one (rows, cards)
bool matrix; every true cell is one output row, written in one call.

FAN-OUT STREAMING: one game writes one row per distinct pool card
(zero rows for an empty pool), not one row per game. The alternative,
one parallel-list row per game, was considered and not taken.

NOT A DECK: this metric's identity unit is the per-game
(draft_id, match_number, game_number) triple plus a pool_card_uuid -
NOT a deck_uuid. The pool here is deck_<name> UNION sideboard_<name>, a
different card multiset than any GenericDeck this container mints
(ChunkDecks hashes only deck_<name>), so this class takes no deck_box
and never reads chunk.decks.
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

import numpy as np
import pyarrow as pa

from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/tutor_target_pool.parquet"
)

_OUTPUT_SCHEMA = pa.schema(
    [
        ("draft_id", pa.string()),
        ("match_number", pa.int64()),
        ("game_number", pa.int64()),
        ("pool_card_uuid", pa.string()),
        ("tutored", pa.bool_()),
    ]
)


class TutorTargetPoolMetric:
    """One game's full draft pool -> per-pool-card tutored bool, one
    output row per (game, pool card) pair.

    Satisfies the Metric[GameDataChunk] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path | None = None,
    ) -> None:
        """Open the output for streaming.

        Inputs:
            version_metadata: the CardBinder version this run reads,
                stamped onto the output.
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
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        output_schema = schema_with_version_metadata(_OUTPUT_SCHEMA, version_metadata)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(self._output_path, output_schema)

    def accumulate(self, chunk: GameDataChunk) -> None:
        """Write one output row per (game, distinct pool card) of chunk.

        Inputs: chunk.
        Output: none.
        Side effects: writes every (row, pool card) pair of this chunk,
            row-major, to the open ParquetBuilder (none if every pool
            is empty).
        Exceptions: none expected.

        Example:
            >>> metric = TutorTargetPoolMetric(version_metadata)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        deck = chunk.zones[GameZone.DECK]
        sideboard = chunk.zones[GameZone.SIDEBOARD]

        # One card axis: the distinct cards of the deck and sideboard
        pool_cards = _distinct_cards(deck.card_uuids + sideboard.card_uuids)
        in_pool = deck.present_for(pool_cards) | sideboard.present_for(pool_cards)
        tutored = chunk.zones[GameZone.TUTORED].present_for(pool_cards)

        # Every pool cell is one output row
        rows, cards = np.nonzero(in_pool)
        card_uuid_strings = np.array([str(card) for card in pool_cards], object)
        self._writer.write_columns(
            {
                "draft_id": chunk.keys.draft_id[rows],
                "match_number": chunk.keys.match_number[rows],
                "game_number": chunk.keys.game_number[rows],
                "pool_card_uuid": card_uuid_strings[cards],
                "tutored": tutored[rows, cards],
            }
        )

    def finalize(self) -> Path:
        """Flush any buffered rows and close the underlying writer.

        A true no-op relative to data - every row this instance will
        ever write was already handed over by accumulate(). Idempotent:
        a second call is a no-op.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/tutor_target_pool.parquet')
        """
        self._writer.close()
        return self._output_path


def _distinct_cards(card_uuids: tuple[UUID, ...]) -> tuple[UUID, ...]:
    """card_uuids without repeats, in first-seen order.

    Inputs: card_uuids. Output: tuple of distinct uuids.
    Side effects: none. Exceptions: none.
    """
    return tuple(dict.fromkeys(card_uuids))
