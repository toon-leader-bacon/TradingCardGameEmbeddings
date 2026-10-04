"""Concrete ReplayTurnEventRateMetric (replay_turn_event_rate_metric.py)
subclasses: the two combat rates.
"""

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    ReplayDataChunk,
    ReplayField,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_turn_event_rate_metric import (
    ReplayTurnEventRateMetric,
)

# Above every card code (int32), so half-turn * _CODE_SPAN + code is one
# int64 key per (half-turn, card) pair
_CODE_SPAN = 2**31


class CombatKillInvolvementRateMetric(ReplayTurnEventRateMetric):
    """Card -> P(a creature died in combat that half-turn | card fought
    that half-turn).

    Not "this card died" - "a creature (either side's) died as a result
    of a half-turn this card fought in". The denominator is each
    half-turn's distinct matched cards in creatures_attacked union
    creatures_blocking; a half-turn is a hit for all of them when either
    side's creatures_killed_combat names anything, matched or not (a
    creature not in the binder still died).
    """

    OUTPUT_STEM: ClassVar[str] = "combat_kill_involvement_rate"
    LABEL_COLUMN: ClassVar[str] = "combat_kill_involvement_rate"

    def _denominator_and_hit_codes(
        self, chunk: ReplayDataChunk
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """See ReplayTurnEventRateMetric._denominator_and_hit_codes(): each
        half-turn's distinct fighters, and those of half-turns with any
        combat kill (matched or not).

        Inputs: chunk. Output: (denominator codes, hit codes).
        Side effects: none. Exceptions: none.
        """
        half_turns, codes = _distinct_fighters(chunk)
        killed = chunk.events_for(
            (
                ReplayField.USER_CREATURES_KILLED_COMBAT,
                ReplayField.OPPO_CREATURES_KILLED_COMBAT,
            )
        )
        return codes, codes[np.isin(half_turns, killed.half_turn_ids())]


class CombatDamagePushThroughRateMetric(ReplayTurnEventRateMetric):
    """Card -> P(card in creatures_unblocked | card in creatures_attacked,
    same half-turn) - how often this attacker's damage connects.

    The denominator counts every matched attacked entry (two copies
    attacking count twice); a hit is each distinct (half-turn, card) that
    attacked and was unblocked, as the row implementation counted.
    """

    OUTPUT_STEM: ClassVar[str] = "combat_damage_push_through_rate"
    LABEL_COLUMN: ClassVar[str] = "combat_damage_push_through_rate"

    def _denominator_and_hit_codes(
        self, chunk: ReplayDataChunk
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """See ReplayTurnEventRateMetric._denominator_and_hit_codes(): every
        matched creatures_attacked entry, and each distinct (half-turn,
        card) both attacked and unblocked.

        Inputs: chunk. Output: (denominator codes, hit codes).
        Side effects: none. Exceptions: none.
        """
        attacked = chunk.events[ReplayField.CREATURES_ATTACKED].matched()
        unblocked = chunk.events[ReplayField.CREATURES_UNBLOCKED].matched()
        attacked_keys, attacked_codes = _distinct_pairs(
            attacked.half_turn_ids(), attacked.codes
        )
        unblocked_keys, _ = _distinct_pairs(unblocked.half_turn_ids(), unblocked.codes)
        hit = np.isin(attacked_keys, unblocked_keys)
        return attacked.codes, attacked_codes[hit]


def _distinct_fighters(
    chunk: ReplayDataChunk,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """Each half-turn's distinct matched cards in creatures_attacked union
    creatures_blocking.

    Inputs: chunk.
    Output: (half-turn ids, codes), one entry per distinct (half-turn,
        card).
    Side effects: none. Exceptions: none.
    """
    fighters = chunk.events_for(
        (ReplayField.CREATURES_ATTACKED, ReplayField.CREATURES_BLOCKING)
    ).matched()
    keys, codes = _distinct_pairs(fighters.half_turn_ids(), fighters.codes)
    return keys // _CODE_SPAN, codes


def _distinct_pairs(
    half_turns: npt.NDArray[np.int64], codes: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """The distinct (half-turn, card) pairs among entries, as one int64
    key per pair (half-turn * _CODE_SPAN + code), sorted, and each
    pair's code.

    Inputs: half_turns, codes (matched, >= 0), one per entry.
    Output: (pair keys, codes), one per distinct pair.
    Side effects: none. Exceptions: none.
    """
    keys = np.unique(half_turns * _CODE_SPAN + codes)
    return keys, (keys % _CODE_SPAN).astype(np.int32)
