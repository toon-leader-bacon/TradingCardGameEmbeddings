"""Each translated game's raw-rarity -> RarityTier table, and the
translators built from them (RARITY_TRANSLATORS).

Data, not logic: tuning a game's mapping is an edit here. Tier meaning
(see src/schema/rarity_tier.py): TIER_1 most common ... TIER_4 rarest;
SPECIAL is scarce but off the ladder; OTHER is not a rarity.

Dominion has no rarity and no translator, so it gets no rows. Pokemon and
MTG tables assume the binder stores a card's lowest-print rarity (see the
ingestion stages in src/data_refinement/card_binder/).
"""

from typing import Mapping

from src.data_refinement.metrics.cross_game.rarity.rarity_translator import (
    FieldRarityTranslator,
    RarityTranslator,
)
from src.schema.game_id import GameId
from src.schema.rarity_tier import RarityTier

_T1, _T2, _T3, _T4 = (
    RarityTier.TIER_1,
    RarityTier.TIER_2,
    RarityTier.TIER_3,
    RarityTier.TIER_4,
)
_SPECIAL, _OTHER = RarityTier.SPECIAL, RarityTier.OTHER

_MTG_TIERS: Mapping[str, RarityTier] = {
    "common": _T1,
    "uncommon": _T2,
    "rare": _T3,
    "mythic": _T4,
    "special": _SPECIAL,
    "bonus": _SPECIAL,
}

_HEARTHSTONE_TIERS: Mapping[str, RarityTier] = {
    "FREE": _T1,
    "COMMON": _T1,
    "RARE": _T2,
    "EPIC": _T3,
    "LEGENDARY": _T4,
}

_GWENT_TIERS: Mapping[str, RarityTier] = {
    "common": _T1,
    "rare": _T2,
    "epic": _T3,
    "legendary": _T4,
}

_FLESH_AND_BLOOD_TIERS: Mapping[str, RarityTier] = {
    "basic": _T1,
    "common": _T1,
    "rare": _T2,
    "super-rare": _T3,
    "majestic": _T3,
    "legendary": _T4,
    "fabled": _T4,
    "promo": _SPECIAL,
    "promo-marvel": _SPECIAL,
    "marvel": _SPECIAL,
    "token": _OTHER,
}

_SLAY_THE_SPIRE_2_TIERS: Mapping[str, RarityTier] = {
    "Basic": _T1,
    "Common": _T1,
    "Uncommon": _T2,
    "Rare": _T3,
    "Ancient": _T4,
    "Event": _SPECIAL,
    "Curse": _OTHER,
    "Status": _OTHER,
    "Token": _OTHER,
    "Quest": _OTHER,
}

# Every ultra / secret / illustration / shiny / ex / V / GX / ACE SPEC /
# LEGEND variant is one top tier: the gaps between them are small
_POKEMON_TOP_TIER_RARITIES = (
    "Rare Ultra",
    "Illustration Rare",
    "Ultra Rare",
    "Double Rare",
    "Rare Secret",
    "Rare Rainbow",
    "Rare Holo EX",
    "Rare Holo V",
    "Special Illustration Rare",
    "Rare Holo GX",
    "Rare Shiny",
    "Shiny Rare",
    "Rare Holo VMAX",
    "Trainer Gallery Rare Holo",
    "Hyper Rare",
    "Rare Holo LV.X",
    "Rare Holo VSTAR",
    "Rare Shiny GX",
    "ACE SPEC Rare",
    "Rare BREAK",
    "Rare Prime",
    "Rare Prism Star",
    "Rare Holo Star",
    "LEGEND",
    "Rare Shining",
    "Radiant Rare",
    "Rare ACE",
    "Shiny Ultra Rare",
    "Amazing Rare",
    "Mega Hyper Rare",
    "MEGA_ATTACK_RARE",
    "Futuristic Rare",
    "Black White Rare",
    "Rare Holo ex",
    "Holo Rare V",
    "Holo Rare VMAX",
    "Holo Rare VSTAR",
)

_POKEMON_TIERS: Mapping[str, RarityTier] = {
    "Common": _T1,
    "Uncommon": _T2,
    "Rare": _T3,
    "Rare Holo": _T3,
    **{rarity: _T4 for rarity in _POKEMON_TOP_TIER_RARITIES},
    "Promo": _SPECIAL,
    "Classic Collection": _SPECIAL,
    "Pikachu Rare": _SPECIAL,
}

_TIERS_BY_GAME: Mapping[GameId, Mapping[str, RarityTier]] = {
    GameId.MTG: _MTG_TIERS,
    GameId.HEARTHSTONE: _HEARTHSTONE_TIERS,
    GameId.GWENT: _GWENT_TIERS,
    GameId.FLESH_AND_BLOOD: _FLESH_AND_BLOOD_TIERS,
    GameId.SLAY_THE_SPIRE_2: _SLAY_THE_SPIRE_2_TIERS,
    GameId.POKEMON: _POKEMON_TIERS,
}

# Every game's rarity is the top-level raw_content "rarity" string
RARITY_TRANSLATORS: Mapping[GameId, RarityTranslator] = {
    game: FieldRarityTranslator(game=game, field="rarity", tiers=tiers)
    for game, tiers in _TIERS_BY_GAME.items()
}
