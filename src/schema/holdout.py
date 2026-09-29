"""Card-level holdout: which tier (train/test/validation) each card belongs to.

CardTier is a property of a CARD, deliberately distinct from Split (a view
of ROWS in a dojo's data). A row split says which examples a dojo yields;
a card tier says which cards may be *seen* while building those examples.
The two layer: a TEST-split row is built through a lookup that hides
VALIDATION-tier cards.

HoldoutSpec is pure and deterministic - the tier of a card depends only on
the spec and the card's identity, never on call order or stored state - so
every container (card binder, dojos, evaluation) agrees without sharing
any mutable object. The filtering itself lives in
data_refinement/card_binder/visible_card_lookup.py.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from uuid import UUID

from src.schema.game_id import GameId
from src.schema.splits import Split
from src.schema.ttv_splits import TTVSplits


class CardTier(StrEnum):
    """Which held-out group a card belongs to."""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


# Order matches TTVSplits' (train, test, validate) convention.
_TIER_ORDER = (CardTier.TRAIN, CardTier.TEST, CardTier.VALIDATION)

# Hashes are bucketed into this many integer slots, then carved into tiers
# by TTVSplits.get_split_indices.
_HASH_RESOLUTION = 2**32

# Nested visibility: each view sees its own tier and every lower one.
_VISIBLE_TIERS: dict[Split, frozenset[CardTier]] = {
    Split.TRAIN: frozenset({CardTier.TRAIN}),
    Split.TEST: frozenset({CardTier.TRAIN, CardTier.TEST}),
    Split.VALIDATION: frozenset(CardTier),
}


@dataclass(frozen=True)
class HoldoutSpec:
    """Assigns every card to a CardTier.

    seed: salts the hash so different experiments hold out different cards.
    tier_ratios: relative sizes of (TRAIN, TEST, VALIDATION) card tiers,
        unnormalized in TTVSplits.from_unnormalized's sense: (8, 1, 1) is
        80% / 10% / 10%. Must be non-negative with a positive sum.
    held_out_games: every card of these games is VALIDATION-tier,
        regardless of its hash (tests generalization to unseen games).

    Inputs: see fields.
    Output: n/a.
    Side effects: none.
    Exceptions: ValueError on construction if a ratio is negative or all
        ratios are zero.

    Example:
        >>> spec = HoldoutSpec(seed=0, tier_ratios=(8, 1, 1),
        ...                    held_out_games=frozenset({GameId.GWENT}))
        >>> spec.tier_of(some_uuid, GameId.MTG)
        <CardTier.TRAIN: 'train'>
    """

    seed: int
    tier_ratios: tuple[float, float, float]
    held_out_games: frozenset[GameId]

    def __post_init__(self) -> None:
        if any(ratio < 0 for ratio in self.tier_ratios) or sum(self.tier_ratios) <= 0:
            raise ValueError(
                f"tier_ratios must be non-negative with a positive sum, "
                f"got {self.tier_ratios}"
            )

    @classmethod
    def no_holdout(cls, seed: int = 0) -> "HoldoutSpec":
        """A spec that puts every card in TRAIN (every view sees every card).

        Inputs: seed (int). Output: HoldoutSpec. Side effects: none.
        Exceptions: none.
        """
        return cls(seed=seed, tier_ratios=(1, 0, 0), held_out_games=frozenset())

    @cached_property
    def _tier_boundaries(self) -> list[tuple[int, int]]:
        """[start, end) hash ranges of (TRAIN, TEST, VALIDATION)."""
        splits: TTVSplits[None] = TTVSplits.from_unnormalized(list(self.tier_ratios))
        return splits.get_split_indices(_HASH_RESOLUTION)

    def tier_of(self, nocab_uuid: UUID, source_game: GameId) -> CardTier:
        """Return the tier of one card.

        Inputs: nocab_uuid (UUID) and source_game (GameId) of the card.
        Output: CardTier. A held-out game is always VALIDATION; otherwise
            the tier comes from a stable hash of (seed, uuid), so it is
            the same across processes and machines.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> spec.tier_of(card.nocab_uuid, card.source_game)
            <CardTier.TEST: 'test'>
        """
        if source_game in self.held_out_games:
            return CardTier.VALIDATION
        position = self._hash_position(nocab_uuid)
        for tier, (start, end) in zip(_TIER_ORDER, self._tier_boundaries):
            if start <= position < end:
                return tier
        return CardTier.VALIDATION  # unreachable: boundaries cover the range

    def to_json(self) -> str:
        """Serialize losslessly: from_json(spec.to_json()) == spec.

        Inputs: none.
        Output: str, a JSON object {"seed": int, "tier_ratios": [3 numbers],
            "held_out_games": [GameId values, sorted]}. Sorting makes equal
            specs serialize to identical text.
        Side effects: none. Exceptions: none.

        Example:
            >>> HoldoutSpec.no_holdout().to_json()
            '{"seed": 0, "tier_ratios": [1, 0, 0], "held_out_games": []}'
        """
        payload = {
            "seed": self.seed,
            "tier_ratios": list(self.tier_ratios),
            "held_out_games": sorted(game.value for game in self.held_out_games),
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, text: str) -> "HoldoutSpec":
        """Parse text written by to_json.

        Inputs: text (str).
        Output: HoldoutSpec equal to the one that wrote it.
        Side effects: none.
        Exceptions: ValueError if text is not JSON, is not an object with
            exactly the keys seed / tier_ratios / held_out_games, a value
            has the wrong type (seed not an int, tier_ratios not three
            numbers, a game not a GameId value), or the spec itself is
            invalid (__post_init__).

        Example:
            >>> HoldoutSpec.from_json(spec.to_json()) == spec
            True
        """
        # Parse the JSON text into a dict (json.JSONDecodeError is a ValueError)
        payload = _holdout_fields_of(json.loads(text))
        # Convert each field to its typed form; __post_init__ checks the ratios
        return cls(
            seed=_parse_seed(payload["seed"]),
            tier_ratios=_parse_tier_ratios(payload["tier_ratios"]),
            held_out_games=_parse_held_out_games(payload["held_out_games"]),
        )

    def visible_tiers(self, view: Split) -> frozenset[CardTier]:
        """Return the tiers whose cards a given split's view may see.

        Inputs: view (Split).
        Output: frozenset[CardTier]; TRAIN -> {TRAIN}, TEST -> {TRAIN,
            TEST}, VALIDATION -> all three.
        Side effects: none.
        Exceptions: none.

        Example:
            >>> spec.visible_tiers(Split.TEST)
            frozenset({<CardTier.TRAIN: 'train'>, <CardTier.TEST: 'test'>})
        """
        return _VISIBLE_TIERS[view]

    def _hash_position(self, nocab_uuid: UUID) -> int:
        """Map (seed, uuid) to a stable integer in [0, _HASH_RESOLUTION)."""
        digest = hashlib.sha256(f"{self.seed}:{nocab_uuid}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % _HASH_RESOLUTION


_HOLDOUT_KEYS = frozenset({"seed", "tier_ratios", "held_out_games"})


def _holdout_fields_of(payload: object) -> dict[str, object]:
    """payload narrowed to a dict with exactly _HOLDOUT_KEYS; ValueError
    otherwise (not an object, or missing/extra keys)."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"holdout JSON must be an object, got {type(payload).__name__}"
        )
    keys = set(payload)
    if keys != _HOLDOUT_KEYS:
        raise ValueError(
            f"holdout JSON keys must be {sorted(_HOLDOUT_KEYS)}, got {sorted(keys)}"
        )
    return payload


def _parse_seed(value: object) -> int:
    """value as an int; ValueError if it is not one (bool rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"holdout seed must be an int, got {value!r}")
    return value


def _parse_tier_ratios(value: object) -> tuple[float, float, float]:
    """value as three floats; ValueError unless a list of exactly three
    numbers (bool rejected)."""
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(_is_number(ratio) for ratio in value)
    ):
        raise ValueError(f"holdout tier_ratios must be three numbers, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_held_out_games(value: object) -> frozenset[GameId]:
    """value as a frozenset of GameId; ValueError unless a list of GameId
    values."""
    if not isinstance(value, list):
        raise ValueError(f"holdout held_out_games must be a list, got {value!r}")
    # GameId(...) raises ValueError on an unknown value
    return frozenset(GameId(game) for game in value)


def _is_number(value: object) -> bool:
    """True for an int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
