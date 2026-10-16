"""Holdout: which tier (train/test/validation) each card belongs to, and
which split each deck belongs to.

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

A deck's split is a Split, not a CardTier: a deck is a source of rows for
deck-derived dojos (contrastive), so its assignment says which split's rows
it feeds. It comes from the same kind of stable hash, salted apart from the
card one, so the two are independent draws.
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

# Order matches TTVSplits' (train, test, validate) convention.
_DECK_SPLIT_ORDER = (Split.TRAIN, Split.TEST, Split.VALIDATION)

# Salts the deck hash apart from the card hash (see _hash_position)
_DECK_HASH_PREFIX = "deck"

# Recorded in deck_split_key: bump it if deck hashing ever changes, so every
# index and cache derived from a deck split is rebuilt
_DECK_HASH_SCHEME = 1

# The deck split of a spec built in code or from a config that names none
DEFAULT_DECK_SPLIT_RATIOS = (8.0, 1.0, 1.0)

# Hashes are bucketed into this many integer slots, then carved into tiers
# by TTVSplits.get_split_indices.
_HASH_RESOLUTION = 2**32

# Nested visibility: each view sees its own tier and every lower one.
_VISIBLE_TIERS: dict[Split, frozenset[CardTier]] = {
    Split.TRAIN: frozenset({CardTier.TRAIN}),
    Split.TEST: frozenset({CardTier.TRAIN, CardTier.TEST}),
    Split.VALIDATION: frozenset(CardTier),
}


class UnknownDeckSplitError(ValueError):
    """A deck split was asked of a HoldoutSpec that has none: one read from
    a holdout.json written before decks were split by the spec (those runs
    split decks per dojo, by a ranking no spec can reproduce)."""


@dataclass(frozen=True)
class HoldoutSpec:
    """Assigns every card to a CardTier and every deck to a Split.

    seed: salts the hash so different experiments hold out different cards.
    tier_ratios: relative sizes of (TRAIN, TEST, VALIDATION) card tiers,
        unnormalized in TTVSplits.from_unnormalized's sense: (8, 1, 1) is
        80% / 10% / 10%. Must be non-negative with a positive sum.
    held_out_games: every card of these games is VALIDATION-tier, and
        every deck a VALIDATION deck, regardless of its hash (tests
        generalization to unseen games).
    deck_split_ratios: relative sizes of (TRAIN, TEST, VALIDATION) decks,
        unnormalized like tier_ratios. None means the deck split is
        unknown: only from_json gives it, for a pre-deck-split holdout,
        and deck_split_of / deck_split_key then raise
        UnknownDeckSplitError rather than guess.

    Inputs: see fields.
    Output: n/a.
    Side effects: none.
    Exceptions: ValueError on construction if a ratio (of either set) is
        negative or all of a set's ratios are zero.

    Example:
        >>> spec = HoldoutSpec(seed=0, tier_ratios=(8, 1, 1),
        ...                    held_out_games=frozenset({GameId.GWENT}))
        >>> spec.tier_of(some_uuid, GameId.MTG)
        <CardTier.TRAIN: 'train'>
    """

    seed: int
    tier_ratios: tuple[float, float, float]
    held_out_games: frozenset[GameId]
    deck_split_ratios: tuple[float, float, float] | None = DEFAULT_DECK_SPLIT_RATIOS

    def __post_init__(self) -> None:
        _require_valid_ratios("tier_ratios", self.tier_ratios)
        if self.deck_split_ratios is not None:
            _require_valid_ratios("deck_split_ratios", self.deck_split_ratios)

    @classmethod
    def no_holdout(cls, seed: int = 0) -> "HoldoutSpec":
        """A spec that puts every card in TRAIN (every view sees every card).
        Decks are still split, by the default deck ratios: TEST and
        VALIDATION rows of deck-derived dojos need decks.

        Inputs: seed (int). Output: HoldoutSpec. Side effects: none.
        Exceptions: none.
        """
        return cls(seed=seed, tier_ratios=(1, 0, 0), held_out_games=frozenset())

    @cached_property
    def _tier_boundaries(self) -> list[tuple[int, int]]:
        """[start, end) hash ranges of (TRAIN, TEST, VALIDATION) cards."""
        return _hash_boundaries(self.tier_ratios)

    @cached_property
    def _deck_split_boundaries(self) -> list[tuple[int, int]]:
        """[start, end) hash ranges of (TRAIN, TEST, VALIDATION) decks.
        Exceptions: UnknownDeckSplitError when deck_split_ratios is None."""
        return _hash_boundaries(self._require_deck_split())

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
        position = self._hash_position(f"{self.seed}:{nocab_uuid}")
        return _TIER_ORDER[_bucket_of(position, self._tier_boundaries)]

    def deck_split_of(self, deck_uuid: UUID, source_game: GameId) -> Split:
        """Return the split of one deck.

        Inputs: deck_uuid (UUID) and source_game (GameId) of the deck.
        Output: Split. A held-out game's deck is always VALIDATION;
            otherwise the split comes from a stable hash of (seed,
            deck_uuid), salted apart from the card hash, carved by
            deck_split_ratios.
        Side effects: none.
        Exceptions: UnknownDeckSplitError when deck_split_ratios is None
            (checked before the held-out-game shortcut, so an old spec
            never answers for some decks and not others).

        Example:
            >>> spec.deck_split_of(deck.nocab_uuid, deck.source_game)
            <Split.TRAIN: 'train'>
        """
        # Validate inputs: refuse an unknown deck split for every deck
        self._require_deck_split()

        if source_game in self.held_out_games:
            return Split.VALIDATION
        position = self._hash_position(f"{_DECK_HASH_PREFIX}:{self.seed}:{deck_uuid}")
        return _DECK_SPLIT_ORDER[_bucket_of(position, self._deck_split_boundaries)]

    def deck_split_key(self, source_game: GameId) -> str:
        """Everything deck_split_of depends on for source_game's decks, as
        a stable string: the cache key for anything derived from one
        game's deck split (a dealer's index, a document-frequency cache).
        Holding out a different game, or changing tier_ratios, leaves it
        unchanged; equal-proportion ratios ((8, 1, 1), (16.0, 2.0, 2.0))
        give the same key.

        Inputs: source_game.
        Output: str, JSON: {"held_out": true} for a held-out game (every
            deck VALIDATION, whatever the seed); else the hashing scheme,
            the seed and the normalized deck_split_ratios.
        Side effects: none.
        Exceptions: UnknownDeckSplitError when deck_split_ratios is None.

        Example:
            >>> HoldoutSpec.no_holdout().deck_split_key(GameId.MTG)
            '{"scheme": 1, "seed": 0, "deck_split_fractions": [0.8, 0.1, 0.1]}'
        """
        # Validate inputs: an unknown deck split has no key
        ratios = self._require_deck_split()

        if source_game in self.held_out_games:
            return json.dumps({"held_out": True})
        return json.dumps(
            {
                "scheme": _DECK_HASH_SCHEME,
                "seed": self.seed,
                "deck_split_fractions": _normalized_fractions(ratios),
            }
        )

    def card_tier_key(self) -> str:
        """Everything tier_of depends on, as a stable string: what anything
        fitted on one tier's cards (e.g. whitening stats) must match.
        deck_split_ratios is not in it.

        Inputs: none.
        Output: str, JSON of the seed, normalized tier_ratios and sorted
            held_out_games.
        Side effects: none. Exceptions: none.

        Example:
            >>> HoldoutSpec.no_holdout().card_tier_key()
            '{"seed": 0, "tier_fractions": [1.0, 0.0, 0.0], "held_out_games": []}'
        """
        return json.dumps(
            {
                "seed": self.seed,
                "tier_fractions": _normalized_fractions(self.tier_ratios),
                "held_out_games": sorted(game.value for game in self.held_out_games),
            }
        )

    def to_json(self) -> str:
        """Serialize losslessly: from_json(spec.to_json()) == spec.

        Inputs: none.
        Output: str, a JSON object {"seed": int, "tier_ratios": [3 numbers],
            "held_out_games": [GameId values, sorted], "deck_split_ratios":
            [3 numbers] or null}. Sorting makes equal specs serialize to
            identical text.
        Side effects: none. Exceptions: none.

        Example:
            >>> json.loads(HoldoutSpec.no_holdout().to_json())["deck_split_ratios"]
            [8.0, 1.0, 1.0]
        """
        payload = {
            "seed": self.seed,
            "tier_ratios": list(self.tier_ratios),
            "held_out_games": sorted(game.value for game in self.held_out_games),
            "deck_split_ratios": (
                None if self.deck_split_ratios is None else list(self.deck_split_ratios)
            ),
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, text: str) -> "HoldoutSpec":
        """Parse text written by to_json.

        Inputs: text (str).
        Output: HoldoutSpec equal to the one that wrote it; text without
            deck_split_ratios (written before decks were split by the
            spec) gives deck_split_ratios=None.
        Side effects: none.
        Exceptions: ValueError if text is not JSON, is not an object with
            the keys seed / tier_ratios / held_out_games (and optionally
            deck_split_ratios) and no others, a value has the wrong type
            (seed not an int, tier_ratios or deck_split_ratios not three
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
            tier_ratios=_parse_three_ratios("tier_ratios", payload["tier_ratios"]),
            held_out_games=_parse_held_out_games(payload["held_out_games"]),
            deck_split_ratios=_parse_deck_split_ratios(
                payload.get("deck_split_ratios")
            ),
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

    def _require_deck_split(self) -> tuple[float, float, float]:
        """deck_split_ratios, known. Inputs: none. Output: the ratios.
        Side effects: none. Exceptions: UnknownDeckSplitError when
        deck_split_ratios is None."""
        if self.deck_split_ratios is None:
            raise UnknownDeckSplitError(
                "this holdout has no deck split (it was saved before decks were "
                "split by the holdout spec, so its runs' deck splits are unknown)"
            )
        return self.deck_split_ratios

    @staticmethod
    def _hash_position(hash_input: str) -> int:
        """Map hash_input (seed and uuid, and a salt for decks) to a stable
        integer in [0, _HASH_RESOLUTION). Inputs: hash_input. Output: int.
        Side effects: none. Exceptions: none."""
        digest = hashlib.sha256(hash_input.encode()).digest()
        return int.from_bytes(digest[:8], "big") % _HASH_RESOLUTION


# deck_split_ratios is optional: holdout.json files written before decks were
# split by the spec lack it
_HOLDOUT_KEYS = frozenset({"seed", "tier_ratios", "held_out_games"})
_OPTIONAL_HOLDOUT_KEYS = frozenset({"deck_split_ratios"})


def _require_valid_ratios(name: str, ratios: tuple[float, float, float]) -> None:
    """Inputs: name (for the message), ratios. Output: None. Side effects:
    none. Exceptions: ValueError if a ratio is negative or they sum to
    zero or less."""
    if any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
        raise ValueError(
            f"{name} must be non-negative with a positive sum, got {ratios}"
        )


def _hash_boundaries(ratios: tuple[float, float, float]) -> list[tuple[int, int]]:
    """[start, end) hash ranges carving [0, _HASH_RESOLUTION) by ratios, in
    (train, test, validation) order. Inputs: ratios (valid). Output: three
    ranges. Side effects: none. Exceptions: none."""
    splits: TTVSplits[None] = TTVSplits.from_unnormalized(list(ratios))
    return splits.get_split_indices(_HASH_RESOLUTION)


def _normalized_fractions(ratios: tuple[float, float, float]) -> list[float]:
    """ratios divided by their sum, so equal-proportion ratios give equal
    keys. Inputs: ratios (valid). Output: three floats summing to ~1.
    Side effects: none. Exceptions: none."""
    total = sum(ratios)
    return [ratio / total for ratio in ratios]


def _bucket_of(position: int, boundaries: list[tuple[int, int]]) -> int:
    """Index of the [start, end) range holding position; the last index if
    none does (unreachable: the ranges cover the hash space). Inputs:
    position, boundaries. Output: int. Side effects: none. Exceptions:
    none."""
    for index, (start, end) in enumerate(boundaries):
        if start <= position < end:
            return index
    return len(boundaries) - 1


def _holdout_fields_of(payload: object) -> dict[str, object]:
    """payload narrowed to a dict with every _HOLDOUT_KEYS key and no key
    outside _HOLDOUT_KEYS | _OPTIONAL_HOLDOUT_KEYS; ValueError otherwise
    (not an object, or missing/extra keys). Side effects: none."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"holdout JSON must be an object, got {type(payload).__name__}"
        )
    keys = set(payload)
    if not _HOLDOUT_KEYS <= keys <= _HOLDOUT_KEYS | _OPTIONAL_HOLDOUT_KEYS:
        raise ValueError(
            f"holdout JSON keys must be {sorted(_HOLDOUT_KEYS)}, optionally with "
            f"{sorted(_OPTIONAL_HOLDOUT_KEYS)}, got {sorted(keys)}"
        )
    return payload


def _parse_seed(value: object) -> int:
    """value as an int; ValueError if it is not one (bool rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"holdout seed must be an int, got {value!r}")
    return value


def _parse_three_ratios(name: str, value: object) -> tuple[float, float, float]:
    """value as three floats; ValueError naming name unless a list of
    exactly three numbers (bool rejected). Side effects: none."""
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(_is_number(ratio) for ratio in value)
    ):
        raise ValueError(f"holdout {name} must be three numbers, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_held_out_games(value: object) -> frozenset[GameId]:
    """value as a frozenset of GameId; ValueError unless a list of GameId
    values."""
    if not isinstance(value, list):
        raise ValueError(f"holdout held_out_games must be a list, got {value!r}")
    # GameId(...) raises ValueError on an unknown value
    return frozenset(GameId(game) for game in value)


def _parse_deck_split_ratios(value: object) -> tuple[float, float, float] | None:
    """value as three floats, or None for an absent or null value (a
    pre-deck-split holdout); ValueError unless None or a list of exactly
    three numbers (bool rejected). Side effects: none."""
    if value is None:
        return None
    return _parse_three_ratios("deck_split_ratios", value)


def _is_number(value: object) -> bool:
    """True for an int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
