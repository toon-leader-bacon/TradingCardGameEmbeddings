"""RarityTranslator: one game's rarity field, mapped onto RarityTier.

A translator knows one game's binder schema: which raw_content field
holds the rarity, how each raw value maps to a RarityTier, and which
raw_content paths name or leak the rarity (a dojo masks those). A raw
value missing from the game's table fails loudly, so a new rarity in a
re-download is caught at the next metric run rather than landing in a
wrong class.

FieldRarityTranslator covers every game whose rarity is one top-level
raw_content string (all six today). A game whose rarity needs more than
one field gets its own class satisfying the same Protocol.
"""

from dataclasses import dataclass
from typing import Mapping, Protocol

from src.schema.card import GenericCard
from src.schema.card_factory import FieldPath
from src.schema.game_id import GameId
from src.schema.rarity_tier import RarityTier


class UnmappedRarityError(ValueError):
    """A card carries a raw rarity value its game's table does not list.

    Inputs: game (GameId), raw_value (str). Output: n/a.
    Side effects: none. Exceptions: none.
    """

    def __init__(self, game: GameId, raw_value: str) -> None:
        super().__init__(
            f"{game.value}: no RarityTier for raw rarity {raw_value!r} - "
            "add it to the game's table in translator_tables.py"
        )
        self.game = game
        self.raw_value = raw_value


class RarityTranslator(Protocol):
    """One game's rarity field -> RarityTier."""

    @property
    def game(self) -> GameId:
        """Which game's cards this translates."""
        ...

    def rarity_tier_of(self, card: GenericCard) -> RarityTier | None:
        """The card's tier, or None when it has no rarity value.

        Inputs: card (a card of self.game).
        Output: RarityTier | None.
        Side effects: none.
        Exceptions: UnmappedRarityError for a raw value not in the table.

        Example:
            >>> GWENT_TRANSLATOR.rarity_tier_of(card_with_rarity("epic"))
            <RarityTier.TIER_3: 'tier_3'>
        """
        ...

    def raw_rarity_of(self, card: GenericCard) -> str | None:
        """The card's raw rarity string as the binder stores it, or None
        when it has none.

        Inputs: card. Output: str | None. Side effects: none.
        Exceptions: none.
        """
        ...

    def masked_paths(self) -> tuple[FieldPath, ...]:
        """Every raw_content path that names or leaks the rarity; a dojo
        replaces each with the mask token.

        Inputs: none. Output: tuple of FieldPaths, at least the rarity field itself.
        Side effects: none. Exceptions: none.
        """
        ...


@dataclass(frozen=True)
class FieldRarityTranslator:
    """A RarityTranslator over one top-level raw_content string field.

    game: which game's cards this translates.
    field: the raw_content key holding the rarity ("rarity" for all six
        games today; each matches that game's RarityMaskMetric field).
    tiers: raw rarity string -> RarityTier. Complete for the game: a raw
        value not listed is an error, not a default.
    """

    game: GameId
    field: str
    tiers: Mapping[str, RarityTier]

    def rarity_tier_of(self, card: GenericCard) -> RarityTier | None:
        """See RarityTranslator.rarity_tier_of."""
        raw_rarity = self.raw_rarity_of(card)
        if raw_rarity is None:
            return None

        # A value the table does not list is an error, never a default
        tier = self.tiers.get(raw_rarity)
        if tier is None:
            raise UnmappedRarityError(self.game, raw_rarity)
        return tier

    def raw_rarity_of(self, card: GenericCard) -> str | None:
        """See RarityTranslator.raw_rarity_of. A missing, null or blank
        field means the card has no rarity."""
        return _rarity_text(card.raw_content.get(self.field))

    def masked_paths(self) -> tuple[FieldPath, ...]:
        """See RarityTranslator.masked_paths: just the rarity field."""
        return ((self.field,),)


def _rarity_text(value: object) -> str | None:
    """A raw_content rarity value as text.

    Inputs: value (whatever raw_content holds under the rarity key).
    Output: the stripped string, or None for a missing, null or blank
        value.
    Side effects: none.
    Exceptions: TypeError for a non-string, non-null value (the binder
        should only ever hold a string here).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"a rarity must be a string, got {value!r}")
    return value.strip() or None
