"""AttackerBlockerCombatOutcomeMetric - BRAINSTORM.md's "Attacker Group
vs. Blocker Group -> Combat Outcome": for every half-turn with at least
one matched attacker, group 1 = its creatures_attacked, group 2 = its
creatures_blocking, label = a signed net-kill-count delta.

A vectorized Metric[ReplayDataChunk] and a RowStreamMetric
(../sliced_metric.py), fanning out: one output row per qualifying
half-turn, in (row, user before oppo, turn) order, each group listing
its matched cards in cell order. No deck_box: the identity is
(draft_id, match_number, game_number, actor, turn).

SIGN CONVENTION: net_kill_delta is attacker-favorable-positive - the
number of matched cards in the DEFENDING side's creatures_killed_combat
this half-turn, minus the number in the ATTACKING side's (the actor's
own). Positive means the attacker traded up.
"""

from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    Actor,
    ReplayDataChunk,
    ReplayField,
    TurnEvents,
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
        ("actor", pa.string()),
        ("turn", pa.int64()),
        ("attacker_uuids", pa.list_(pa.string())),
        ("blocker_uuids", pa.list_(pa.string())),
        ("net_kill_delta", pa.int64()),
    ]
)


class AttackerBlockerCombatOutcomeMetric:
    """One attacking half-turn -> (attackers, blockers, net kill delta).

    Satisfies the Metric[ReplayDataChunk] Protocol (../../metric.py) and
    RowStreamMetric (../sliced_metric.py) structurally.
    """

    FAMILY: ClassVar[DataType] = DataType.REPLAY
    OUTPUT_STEM: ClassVar[str] = "attacker_blocker_combat_outcome"
    LABEL_COLUMN: ClassVar[str] = "net_kill_delta"
    IS_ROW_STREAM: ClassVar[Literal[True]] = True

    def __init__(
        self, version_metadata: MetricVersionMetadata, output_path: Path
    ) -> None:
        """Open the output for streaming.

        Inputs: version_metadata (stamped onto the output), output_path
            (this CSV's partition path).
        Output: none (constructor).
        Side effects: creates output_path's parent directories; opens
            output_path for writing through a ParquetBuilder held open
            until finalize().
        Exceptions: whatever ParquetBuilder raises opening output_path.
        """
        self._output_path = output_path
        schema = schema_with_version_metadata(_OUTPUT_SCHEMA, version_metadata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(output_path, schema)

    def accumulate(self, chunk: ReplayDataChunk) -> None:
        """Write one output row per qualifying half-turn of chunk.

        Inputs: chunk.
        Output: none.
        Side effects: writes the chunk's qualifying half-turns to the
            open ParquetBuilder (none if no half-turn attacked).
        Exceptions: none expected.

        Example:
            >>> metric = AttackerBlockerCombatOutcomeMetric(version_metadata, path)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        attackers = chunk.events[ReplayField.CREATURES_ATTACKED].matched()
        blockers = chunk.events[ReplayField.CREATURES_BLOCKING].matched()

        # Qualifying half-turns: those with a matched attacker, in order
        half_turns = np.unique(attackers.half_turn_ids())
        if half_turns.size == 0:
            return
        rows, actors, turns = attackers.split_half_turn_ids(half_turns)

        # Each half-turn's groups and its net kill delta
        card_strings = tuple(str(card_uuid) for card_uuid in chunk.card_uuids)
        self._writer.write_columns(
            {
                "draft_id": chunk.keys.draft_id[rows],
                "match_number": chunk.keys.match_number[rows],
                "game_number": chunk.keys.game_number[rows],
                "actor": np.array([Actor(a).label for a in actors], np.object_),
                "turn": turns,
                "attacker_uuids": _uuid_lists_per_half_turn(
                    attackers, half_turns, card_strings
                ),
                "blocker_uuids": _uuid_lists_per_half_turn(
                    blockers, half_turns, card_strings
                ),
                "net_kill_delta": _net_kill_deltas(chunk, half_turns, actors),
            }
        )

    def finalize(self) -> Path:
        """Flush and close the writer. Idempotent.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetBuilder.
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/replay_data/attacker_blocker_combat_outcome/PIO/TradSealed.parquet')
        """
        self._writer.close()
        return self._output_path


def _net_kill_deltas(
    chunk: ReplayDataChunk,
    half_turns: npt.NDArray[np.int64],
    actors: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Per qualifying half-turn, the defender's matched combat kills
    minus the attacker's own: on a USER half-turn, oppo's killed field
    minus user's; on an OPPO half-turn, the reverse.

    Inputs: chunk, half_turns (sorted, distinct), actors (each half-turn's
        Actor value, from split_half_turn_ids).
    Output: int64 array, one per half-turn.
    Side effects: none. Exceptions: none.
    """
    user_killed = chunk.events[ReplayField.USER_CREATURES_KILLED_COMBAT].matched()
    oppo_killed = chunk.events[ReplayField.OPPO_CREATURES_KILLED_COMBAT].matched()
    user_losses = _counts_per_half_turn(user_killed, half_turns)
    oppo_losses = _counts_per_half_turn(oppo_killed, half_turns)

    # The defender is whoever is not attacking this half-turn
    attacker_is_user = actors == Actor.USER.value
    return np.where(
        attacker_is_user, oppo_losses - user_losses, user_losses - oppo_losses
    )


def _uuid_lists_per_half_turn(
    events: TurnEvents,
    half_turns: npt.NDArray[np.int64],
    card_uuids: tuple[str, ...],
) -> pa.ListArray:
    """For each half-turn id in half_turns (sorted, distinct), the uuids
    of events' entries in that half-turn, in entry order.

    Inputs: events (matched), half_turns, card_uuids (the code table
        as strings).
    Output: pa.ListArray of strings, one list per half-turn (empty where
        none).
    Side effects: none. Exceptions: none.
    """
    positions = _half_turn_positions(events, half_turns)
    in_half_turns = positions >= 0

    # Group the entries by half-turn; a stable sort keeps entry order
    order = np.argsort(positions[in_half_turns], kind="stable")
    codes = events.codes[in_half_turns][order]
    sizes = np.bincount(positions[in_half_turns], minlength=half_turns.shape[0])
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    uuids = pa.array([card_uuids[code] for code in codes], pa.string())
    return pa.ListArray.from_arrays(pa.array(offsets), uuids)


def _counts_per_half_turn(
    events: TurnEvents, half_turns: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """For each half-turn id in half_turns (sorted, distinct), how many of
    events' entries fall in it.

    Inputs: events (matched), half_turns.
    Output: int64 array, one per half-turn.
    Side effects: none. Exceptions: none.
    """
    positions = _half_turn_positions(events, half_turns)
    return np.bincount(positions[positions >= 0], minlength=half_turns.shape[0]).astype(
        np.int64
    )


def _half_turn_positions(
    events: TurnEvents, half_turns: npt.NDArray[np.int64]
) -> npt.NDArray[np.intp]:
    """Each entry's index into half_turns (sorted, distinct), or -1 where
    its half-turn is not listed.

    Inputs: events, half_turns.
    Output: intp array, one per entry.
    Side effects: none. Exceptions: none.
    """
    ids = events.half_turn_ids()
    if half_turns.size == 0:
        return np.full(ids.shape, -1, np.intp)
    positions = np.searchsorted(half_turns, ids)
    clipped = np.minimum(positions, half_turns.shape[0] - 1)
    listed = (positions < half_turns.shape[0]) & (half_turns[clipped] == ids)
    return np.where(listed, positions, -1)
