"""MTGTop8's formats, as its search form names them (the `format`
field of /search), in crawl priority order: eternal formats first,
since they hold the old cards the MTG teacher lacks (see
plans/mtg_deck_sources.md).

The search form also offers COMF and NCOMF, which group other formats;
they are left out so no deck is listed twice.
"""

from enum import StrEnum


class MtgTop8Format(StrEnum):
    """One MTGTop8 format code. Declaration order is the order
    `--mtgtop8-format all` crawls them in. Deck counts from the search
    page on 2026-11-11."""

    VINTAGE = "VI"  # 25,179
    LEGACY = "LE"  # 97,477
    PAUPER = "PAU"  # 81,436
    PREMODERN = "PREM"  # 30,068
    PEASANT = "PEA"  # 3,690
    HIGHLANDER = "HIGH"  # 1,606
    CANADIAN_HIGHLANDER = "CHL"  # 366
    EXTENDED = "EX"  # 2,199
    BLOCK = "BL"  # 3,403
    HISTORIC = "HI"  # 4,867
    EXPLORER = "EXP"  # 2,462
    DUEL_COMMANDER = "EDH"  # 44,254
    CEDH = "cEDH"  # 39,656
    MTGO_COMMANDER = "EDHM"  # 2,232
    COMMANDER_PRECONS = "EDHP"  # 440
    MODERN = "MO"  # 132,530
    PIONEER = "PI"  # 52,442
    STANDARD = "ST"  # 107,231
    LIMITED = "LI"  # 2,671
    ALCHEMY = "ALCH"  # 648
