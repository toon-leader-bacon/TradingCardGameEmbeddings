"""DeckOccurrenceCountMetric - how many distinct drafts chose each deck
this family has seen.

A vectorized Metric[GameDataChunk] (see game_data/README.md). Per
chunk it adds each row's draft_id to the running per-deck_uuid set of
distinct drafts seen so far - no DeckBox of its own. The canonical
DeckBox (src/data_refinement/deck_box/seventeenlands_game_data/
extraction_stage.py) already holds every deck this metric would ever
see, under the same deck_uuid_from_cards() identity.

ACCUMULATION, NOT STREAMING: like
on_play_win_rate_sensitivity_by_deck_metric.py's
OnPlayWinRateSensitivityByDeckMetric, this metric's output needs every
game across the whole scan (a deck's occurrence count can only grow as
more chunks/CSVs arrive), so it does NOT subclass
game_deck_label_metric.py's GameDeckLabelMetric, which writes one row
per game as soon as accumulate() sees it.

WHY DISTINCT draft_id, NOT ROW COUNT: a single draft plays 3-7 games
(see game_deck_label_metric.py's module docstring's PER-GAME
IDENTIFIER section and ChunkDecks/GameKeys), all sharing one deck
(modulo sideboard swaps the row implementation doesn't distinguish -
see chunk_decks.py). Counting rows would inflate a deck's popularity
by however many games each of its drafts happened to play; counting
distinct draft_ids answers the actual question this metric exists for
("how many drafts played this exact decklist") instead.
"""

import dataclasses
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa

from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/deck_occurrence_count.parquet"
)

_OUTPUT_SCHEMA = pa.schema(
    [
        ("deck_uuid", pa.string()),
        ("occurrence_count", pa.int64()),
    ]
)


class DeckOccurrenceCountMetric:
    """Deck -> the number of distinct drafts that chose it, across the
    whole scan.

    Satisfies the Metric[GameDataChunk] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path | None = None,
    ) -> None:
        """Start a metric with no decks seen yet.

        Inputs:
            version_metadata: the CardBinder version this run reads;
                stamped onto the output with requires_deck_box=True
                (a downstream reader of deck_uuid still needs some
                DeckBox - see module docstring).
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none (no I/O until accumulate()/finalize()).
        Exceptions: none.
        """
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._version_metadata = dataclasses.replace(
            version_metadata, requires_deck_box=True
        )
        self._draft_ids_by_deck: dict[UUID, set[str]] = {}

    def accumulate(self, chunk: GameDataChunk) -> None:
        """Add each row's draft_id to its deck's running set of distinct
        drafts.

        Inputs: chunk.
        Output: none.
        Side effects: grows the per-deck_uuid draft_id sets held in
            memory for the lifetime of this instance.
        Exceptions: none.

        Example:
            >>> metric = DeckOccurrenceCountMetric(version_metadata)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        deck_uuids = chunk.decks.row_deck_uuids()
        draft_ids = chunk.keys.draft_id
        for deck_uuid, draft_id in zip(deck_uuids, draft_ids):
            drafts = self._draft_ids_by_deck.setdefault(UUID(str(deck_uuid)), set())
            drafts.add(str(draft_id))

    def finalize(self) -> Path:
        """Write one output row per distinct deck_uuid seen, with its
        occurrence_count, and close the writer.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; opens, writes and closes a ParquetBuilder at
            self._output_path (a parquet file with columns deck_uuid:
            str, occurrence_count: int64 - one row per deck seen at
            least once, in first-seen order).
        Exceptions: whatever ParquetBuilder raises on failure to open or
            write self._output_path.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/deck_occurrence_count.parquet')
        """
        output_schema = schema_with_version_metadata(
            _OUTPUT_SCHEMA, self._version_metadata
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = ParquetBuilder(self._output_path, output_schema)
        for deck_uuid, draft_ids in self._draft_ids_by_deck.items():
            writer.write_row(
                {
                    "deck_uuid": str(deck_uuid),
                    "occurrence_count": len(draft_ids),
                }
            )
        writer.close()
        return self._output_path
