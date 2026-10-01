"""GroupSwapMod - train-only augmentation for symmetric pair tasks in
the multi-group-binary-classification cell.

A pair metric such as "did deck A beat deck B" writes its two decks in a
fixed (uuid-sorted) order. That order carries no information, but a head
that concatenates [group_0, group_1] can still learn an order-specific
function. Swapping the groups and flipping the label (y -> 1 - y) with
probability swap_probability shows TRAIN both orders, pushing the head
toward P(A beats B) = 1 - P(B beats A).

Only valid when the label is "group 0 vs group 1" and both groups are
the same kind of thing; never use it for asymmetric inputs such as
[partial deck, kingdom]. Lives beside its only consumer, this cell.
Never mutates its input (src/dojos/README.md, mods/).
"""

import random
from typing import cast

from src.dojos.mods.mod import Mod
from src.schema.type_hints import MultiGroupInput, TrainingDatum


class GroupSwapMod(Mod):
    """With probability swap_probability, ([g0, g1], y) -> ([g1, g0], 1 - y)."""

    def __init__(
        self,
        swap_probability: float = 0.5,
        rng_seed: int | None = None,
    ) -> None:
        """
        Inputs:
            swap_probability: chance each datum is swapped, in [0, 1].
            rng_seed: seeds this mod's own random.Random; None means
                unseeded.
        Output: none (constructor). Always train_only (augmentation).
        Side effects: none.
        Exceptions: ValueError if swap_probability is outside [0, 1].

        Example:
            >>> ModPipeline([GroupSwapMod(rng_seed=0)])
        """
        super().__init__(train_only=True)
        if not 0.0 <= swap_probability <= 1.0:
            raise ValueError(
                f"swap_probability must lie in [0, 1], got {swap_probability}"
            )
        self.swap_probability = swap_probability
        self._rng = random.Random(rng_seed)

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        """The datum swapped (new outer list, flipped label) or unchanged.

        Inputs: data, a ([group_0, group_1], label) datum, both groups
            non-empty, label a float in {0.0, 1.0}.
        Output: ([group_1, group_0], 1.0 - label) with probability
            swap_probability, else data itself.
        Side effects: advances this mod's own random state; data is
            never modified.
        Exceptions: TypeError if data is not two card lists with a float
            label; ValueError if either group is empty (a symmetric pair
            never has an empty side, so that is malformed data).

        Example:
            >>> GroupSwapMod(1.0).apply_single(([[card_a], [card_b]], 1.0))
            ([[card_b], [card_a]], 0.0)
        """
        groups, label = _split_pair_datum(data)
        if self._rng.random() >= self.swap_probability:
            return data

        # A new outer list; the group lists themselves are shared, not copied
        swapped: MultiGroupInput = [groups[1], groups[0]]
        return (swapped, 1.0 - label)


def _split_pair_datum(data: TrainingDatum) -> tuple[MultiGroupInput, float]:
    """data's two groups and float label, typed and checked.

    Inputs: data (TrainingDatum).
    Output: (groups, label).
    Side effects: none.
    Exceptions: TypeError if the input is not a list of exactly two
        lists or the label is not a float; ValueError if a group is
        empty.
    """
    groups, label = data
    if not (
        isinstance(groups, list)
        and len(groups) == 2
        and all(isinstance(group, list) for group in groups)
    ):
        raise TypeError("GroupSwapMod expects a [group_0, group_1] input")
    if not isinstance(label, float):
        raise TypeError(f"GroupSwapMod expects a float label, got {type(label)}")
    if not groups[0] or not groups[1]:
        raise ValueError("GroupSwapMod expects two non-empty groups")
    return cast(MultiGroupInput, groups), label  # list-of-lists checked above
