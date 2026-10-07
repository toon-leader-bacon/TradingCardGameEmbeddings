"""RarityTier: a card's scarcity on one ladder shared by every game.

Lives in schema/ beside GameId: the cross-game rarity metric, its dojo
and evaluation all read it, and no game's container owns it. Each game's
own rarity strings are mapped onto it by a translator
(src/data_refinement/metrics/cross_game/rarity/rarity_translator.py).
"""

from enum import Enum


class RarityTier(str, Enum):
    """A card's scarcity on the shared ladder.

    TIER_1 is the most common and TIER_4 the rarest. SPECIAL and OTHER
    sit off the ladder.

    TIER_1: the bottom - common, including basic / free / starter cards.
    TIER_2: lower middle.
    TIER_3: upper middle.
    TIER_4: the rarest tier a game has (a game that splits its top finely
        merges it here).
    SPECIAL: scarce but off the ladder - promos, event cards.
    OTHER: not a rarity at all, though the source stores it in the same
        field - curses, statuses, tokens, quests.

    Inputs: none (enum). Output: n/a. Side effects: none. Exceptions: none.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    SPECIAL = "special"
    OTHER = "other"


# The ladder proper, most common first
LADDER: tuple[RarityTier, ...] = (
    RarityTier.TIER_1,
    RarityTier.TIER_2,
    RarityTier.TIER_3,
    RarityTier.TIER_4,
)
