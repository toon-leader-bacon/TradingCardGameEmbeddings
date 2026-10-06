"""DeckOccurrenceCountMetric - how many distinct drafts chose each deck.

Enriches the canonical DeckBox (src/data_refinement/deck_box/
seventeenlands_game_data/), which holds each decklist once, with its
popularity: a later sampler can weight decks by it. No DeckBox of its
own; deck_uuids use the canonical box's deck_uuid_from_cards() identity
(see game_deck_label_metric.py's KNOWN GAP on box coverage).

A vectorized Metric[GameDataChunk] (see game_data/README.md) and a
CountTableMetric (../sliced_metric.py): each partition holds, per
deck_uuid, the number of distinct draft_ids in that CSV that played it.
A draft lives in exactly one CSV (one set and format), so summing the
partitions of a slice counts each draft once: the slice's count is exact.

WHY DISTINCT draft_id, NOT ROW COUNT: a single draft plays 3-7 games
(see game_deck_label_metric.py's module docstring's PER-GAME
IDENTIFIER section), usually all with one deck. Counting rows would
inflate a deck's popularity by however many games each of its drafts
happened to play; counting distinct draft_ids answers "how many drafts
played this exact decklist" instead.
"""

import dataclasses
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import numpy as np
import pandas as pd
import pyarrow as pa

from src.data_refinement.metrics.seventeenlands.count_table import (
    SAMPLE_COUNT_COLUMN,
    write_count_table,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
)
from src.data_refinement.metrics.version_metadata import MetricVersionMetadata
from src.data_retrieval.seventeenlands.refs import DataType

_COUNT_COLUMN = "draft_count"


class DeckOccurrenceCountMetric:
    """Deck -> the number of distinct drafts that played it.

    Satisfies the Metric[GameDataChunk] Protocol (../../metric.py) and
    CountTableMetric (../sliced_metric.py) structurally.
    """

    FAMILY: ClassVar[DataType] = DataType.GAME
    OUTPUT_STEM: ClassVar[str] = "deck_occurrence_count"
    LABEL_COLUMN: ClassVar[str] = "occurrence_count"
    LABEL_VERSION: ClassVar[int] = 1
    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ("deck_uuid",)
    COUNT_COLUMNS: ClassVar[tuple[str, ...]] = (_COUNT_COLUMN,)
    HAS_BASELINE: ClassVar[bool] = False

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path,
    ) -> None:
        """Start a metric with no decks seen yet.

        Inputs:
            version_metadata: the CardBinder version this run reads;
                stamped onto the output with requires_deck_box=True (a
                reader of deck_uuid needs the canonical DeckBox).
            output_path: this CSV's partition path
                (SeventeenLandsPartition.path()).
        Output: none (constructor).
        Side effects: none (no I/O until finalize()).
        Exceptions: none.
        """
        self._output_path = output_path
        self._version_metadata = dataclasses.replace(
            version_metadata, requires_deck_box=True
        )
        self._draft_ids_by_deck: dict[UUID, set[str]] = {}

    def accumulate(self, chunk: GameDataChunk) -> None:
        """Add each row's draft_id to its deck's set of distinct drafts.

        Inputs: chunk.
        Output: none.
        Side effects: grows the per-deck_uuid draft_id sets held in
            memory for the lifetime of this instance (one Python step per
            distinct (deck, draft) pair in the chunk, not per row).
        Exceptions: none.

        Example:
            >>> metric = DeckOccurrenceCountMetric(version_metadata, partition_path)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        # Distinct (deck, draft) pairs first: a draft's games mostly share
        # a deck, so far fewer pairs than rows reach the Python loop
        pairs = pd.DataFrame(
            {"deck": chunk.decks.row_deck, "draft_id": chunk.keys.draft_id}
        ).drop_duplicates()
        for deck_index, draft_id in zip(pairs["deck"], pairs["draft_id"]):
            deck_uuid = chunk.decks.decks[deck_index].nocab_uuid
            self._draft_ids_by_deck.setdefault(deck_uuid, set()).add(str(draft_id))

    def finalize(self) -> Path:
        """Write this partition's per-deck distinct-draft counts.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: writes self._output_path via write_count_table:
            deck_uuid and draft_count (int64), one row per deck seen.
        Exceptions: whatever the parquet write raises.

        Example:
            >>> metric.finalize()
            Path('data/metrics/seventeenlands/game_data/deck_occurrence_count/KTK/TradDraft.parquet')
        """
        deck_uuids = [str(deck_uuid) for deck_uuid in self._draft_ids_by_deck]
        counts = np.array(
            [len(drafts) for drafts in self._draft_ids_by_deck.values()], np.int64
        )
        return write_count_table(
            self._output_path,
            type(self),
            {"deck_uuid": deck_uuids},
            {_COUNT_COLUMN: counts},
            self._version_metadata,
        )

    @classmethod
    def output_from_counts(
        cls, summed: pa.Table, baseline: pa.Table | None
    ) -> pa.Table:
        """See CountTableMetric.output_from_counts(): the label is the
        summed distinct-draft count, which is also the sample count.

        Inputs: summed (deck_uuid + draft_count), baseline (None).
        Output: deck_uuid, occurrence_count (int64), sample_count (int64).
        Side effects: none. Exceptions: none.

        Example:
            >>> DeckOccurrenceCountMetric.output_from_counts(summed, None)
        """
        counts = summed.column(_COUNT_COLUMN)
        result = summed.select(list(cls.KEY_COLUMNS))
        result = result.append_column(cls.LABEL_COLUMN, counts)
        return result.append_column(SAMPLE_COUNT_COLUMN, counts)
