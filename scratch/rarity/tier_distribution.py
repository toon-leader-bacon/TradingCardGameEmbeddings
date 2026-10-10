"""Scratch: % of each game's cards per proposed cross-game rarity tier.

Tiers: T1 (bottom) .. T4 (top), SPECIAL (scarce, off the ladder), OTHER
(not a rarity: curse, token, ...), NONE (no rarity value). Pokemon is
estimated from the raw set files, since its binder prunes rarity: cards
are grouped by the binder's identity (name, set code) and take their
lowest-tier print. Run from the project root with PYTHONPATH=.
"""

import glob
import json
from collections import Counter, defaultdict

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.game_id import GameId

COLUMNS = ("T1", "T2", "T3", "T4", "SPECIAL", "OTHER", "NONE")
# For "lowest print": a ladder tier beats SPECIAL, which beats OTHER/NONE
_PRINT_PREFERENCE = {name: rank for rank, name in enumerate(COLUMNS)}

MTG = {"common": "T1", "uncommon": "T2", "rare": "T3", "mythic": "T4",
       "special": "SPECIAL", "bonus": "SPECIAL"}
HEARTHSTONE = {"FREE": "T1", "COMMON": "T1", "RARE": "T2", "EPIC": "T3",
               "LEGENDARY": "T4"}
GWENT = {"common": "T1", "rare": "T2", "epic": "T3", "legendary": "T4"}
FAB = {"basic": "T1", "common": "T1", "rare": "T2", "super-rare": "T3",
       "majestic": "T3", "legendary": "T4", "fabled": "T4", "token": "OTHER",
       "promo": "SPECIAL", "promo-marvel": "SPECIAL", "marvel": "SPECIAL"}
_STS2_OFF_LADDER = {"Event": "SPECIAL", "Curse": "OTHER", "Status": "OTHER",
                    "Token": "OTHER", "Quest": "OTHER"}
# A: rare at the top, mid-upper empty, Ancient special
STS2_A = {"Basic": "T1", "Common": "T1", "Uncommon": "T2", "Rare": "T4",
          "Ancient": "SPECIAL", **_STS2_OFF_LADDER}
# B: every slot filled, Ancient as the top tier
STS2_B = {"Basic": "T1", "Common": "T1", "Uncommon": "T2", "Rare": "T3",
          "Ancient": "T4", **_STS2_OFF_LADDER}
_POKEMON_TOP = (
    "Rare Ultra", "Illustration Rare", "Ultra Rare", "Double Rare",
    "Rare Secret", "Rare Rainbow", "Rare Holo EX", "Rare Holo V",
    "Special Illustration Rare", "Rare Holo GX", "Rare Shiny", "Shiny Rare",
    "Rare Holo VMAX", "Trainer Gallery Rare Holo", "Hyper Rare",
    "Rare Holo LV.X", "Rare Holo VSTAR", "Rare Shiny GX", "ACE SPEC Rare",
    "Rare BREAK", "Rare Prime", "Rare Prism Star", "Rare Holo Star", "LEGEND",
    "Rare Shining", "Radiant Rare", "Rare ACE", "Shiny Ultra Rare",
    "Amazing Rare", "Mega Hyper Rare", "MEGA_ATTACK_RARE", "Futuristic Rare",
    "Black White Rare", "Rare Holo ex", "Holo Rare V", "Holo Rare VMAX",
    "Holo Rare VSTAR",
)
POKEMON = {"Common": "T1", "Uncommon": "T2", "Rare": "T3", "Rare Holo": "T3",
           **{name: "T4" for name in _POKEMON_TOP},
           "Promo": "SPECIAL", "Classic Collection": "SPECIAL",
           "Pikachu Rare": "SPECIAL"}


def tier_of(table: dict[str, str], raw: object) -> str:
    if raw is None:
        return "NONE"
    if raw not in table:
        raise KeyError(f"unmapped rarity {raw!r}")
    return table[raw]


def binder_tiers(game: GameId, table: dict[str, str]) -> Counter[str]:
    binder = CardBinder.load([CardBinder.default_output_path(game)])
    return Counter(
        tier_of(table, card.raw_content.get("rarity"))
        for card in binder.all_cards(game)
    )


def pokemon_tiers() -> tuple[Counter[str], Counter[str]]:
    """(card-level tiers by lowest print, the winning raw string counts)."""
    best: dict[tuple[str, str], tuple[int, str]] = {}
    for path in glob.glob("data/raw/pokemon_tcg/cards/**/*.json", recursive=True):
        with open(path, encoding="utf-8") as file:
            for row in json.load(file):
                key = (row["name"], row["id"].rsplit("-", 1)[0])
                raw = row.get("rarity")
                rank = _PRINT_PREFERENCE[tier_of(POKEMON, raw)]
                if key not in best or rank < best[key][0]:
                    best[key] = (rank, raw)
    tiers = Counter(COLUMNS[rank] for rank, _ in best.values())
    raws = Counter(str(raw) for _, raw in best.values())
    return tiers, raws


def print_row(label: str, counts: Counter[str]) -> None:
    total = sum(counts.values())
    cells = "".join(f"{100 * counts[c] / total:8.1f}" for c in COLUMNS)
    print(f"{label:<22}{total:>7}{cells}")


def main() -> None:
    print(f"{'game':<22}{'cards':>7}" + "".join(f"{c:>8}" for c in COLUMNS))
    print_row("mtg", binder_tiers(GameId.MTG, MTG))
    print_row("hearthstone", binder_tiers(GameId.HEARTHSTONE, HEARTHSTONE))
    print_row("gwent", binder_tiers(GameId.GWENT, GWENT))
    print_row("flesh_and_blood", binder_tiers(GameId.FLESH_AND_BLOOD, FAB))
    print_row("sts2 A (rare top)", binder_tiers(GameId.SLAY_THE_SPIRE_2, STS2_A))
    print_row("sts2 B (ancient top)", binder_tiers(GameId.SLAY_THE_SPIRE_2, STS2_B))
    pokemon, raws = pokemon_tiers()
    print_row("pokemon (lowest print)", pokemon)
    print("\npokemon lowest-print raw values:", raws.most_common(15))


if __name__ == "__main__":
    main()
