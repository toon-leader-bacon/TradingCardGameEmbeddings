"""CombatAggressionProfileMetric - BRAINSTORM.md's "Full Deck ->
Combat-Aggression-Profile Prediction": given the user's full
deck_<name> list, predict a combat-tempo scalar from this same game's
half-turns: the average number of matched attackers per user half-turn
that had any (matched) attack, 0.0 for a game without one.

A vectorized Metric[ReplayDataChunk] and a RowStreamMetric
(../sliced_metric.py): one output row per game, its deck referenced by
deck_uuid. Each chunk's decks are identified once by the parser
(ChunkDecks, ../chunk_decks.py) and stored in the family DeckBox with
store_chunk_decks(); the output is stamped requires_deck_box=True.
"""

import dataclasses
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.seventeenlands.chunk_decks import (
    store_chunk_decks,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    Actor,
    ReplayDataChunk,
    ReplayField,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.data_retrieval.seventeenlands.refs import DataType

_OUTPUT_SCHEMA = pa.schema(
    [
        ("draft_id", pa.string()),
        ("match_number", pa.int64()),
        ("game_number", pa.int64()),
        ("deck_uuid", pa.string()),
        ("combat_aggression_profile", pa.float64()),
    ]
)


class CombatAggressionProfileMetric:
    """Deck -> average attackers per attacking user half-turn.

    Satisfies the Metric[ReplayDataChunk] Protocol (../../metric.py) and
    RowStreamMetric (../sliced_metric.py) structurally.
    """

    FAMILY: ClassVar[DataType] = DataType.REPLAY
    OUTPUT_STEM: ClassVar[str] = "combat_aggression_profile"
    LABEL_COLUMN: ClassVar[str] = "combat_aggression_profile"
    IS_ROW_STREAM: ClassVar[Literal[True]] = True

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        deck_box: DeckBox,
        output_path: Path,
    ) -> None:
        """Open the output for streaming.

        Inputs:
            version_metadata: the CardBinder version this run reads;
                stamped onto the output with requires_deck_box=True.
            deck_box: the metrics-private family DeckBox every deck is
                written into (never the published box).
            output_path: this CSV's partition path.
        Output: none (constructor).
        Side effects: creates output_path's parent directories; opens
            output_path for writing through a ParquetBuilder held open
            until finalize().
        Exceptions: whatever ParquetBuilder raises opening output_path.
        """
        self._deck_box = deck_box
        self._output_path = output_path
        schema = schema_with_version_metadata(
            _OUTPUT_SCHEMA,
            dataclasses.replace(version_metadata, requires_deck_box=True),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(output_path, schema)

    def accumulate(self, chunk: ReplayDataChunk) -> None:
        """Store the chunk's decks and write one row per game.

        Inputs: chunk.
        Output: none.
        Side effects: store_chunk_decks(chunk.decks, deck box); writes
            the chunk's rows to the open ParquetBuilder.
        Exceptions: DeckBox's batch-wide errors.

        Example:
            >>> metric = CombatAggressionProfileMetric(version_metadata, box, path)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        store_chunk_decks(chunk.decks, self._deck_box)
        self._writer.write_columns(
            {
                "draft_id": chunk.keys.draft_id,
                "match_number": chunk.keys.match_number,
                "game_number": chunk.keys.game_number,
                "deck_uuid": chunk.decks.row_deck_uuids(),
                "combat_aggression_profile": _aggression_profiles(chunk),
            }
        )

    def finalize(self) -> Path:
        """Flush and close the writer. Idempotent. Does not save the
        deck box (the driver does).

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder.
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/replay_data/combat_aggression_profile/PIO/TradSealed.parquet')
        """
        self._writer.close()
        return self._output_path


def _aggression_profiles(chunk: ReplayDataChunk) -> npt.NDArray[np.float64]:
    """Per row, the mean matched-attacker count over the user's half-turns
    with at least one matched attacker; 0.0 for a row without one.

    Inputs: chunk. Output: float64 array (rows,).
    Side effects: none. Exceptions: none.
    """
    attackers = (
        chunk.events[ReplayField.CREATURES_ATTACKED].matched().for_actor(Actor.USER)
    )
    half_turns, per_half_turn = np.unique(attackers.half_turn_ids(), return_counts=True)
    rows, _, _ = attackers.split_half_turn_ids(half_turns)

    # Mean over each row's attacking half-turns; 0.0 where it had none
    attack_sum = np.bincount(rows, weights=per_half_turn, minlength=len(chunk))
    attack_turns = np.bincount(rows, minlength=len(chunk))
    result = np.zeros(len(chunk), np.float64)
    np.divide(attack_sum, attack_turns, out=result, where=attack_turns > 0)
    return result
