"""The tcgcsv games, as TcgjsonGameSpecs over tcgcsv's catalog
(TCGCSV_FORMAT): their attribute keys are TCGplayer's extendedData names
("Attack Points"), not tcgjson's customAttributes keys ("attackPoints").

Every game here is a Bandai game whose card is its card number (many
cards share a name: "Giblet" is both a Battle card and a Leader), so
each sets card_number. Each drops the store-wide noise: Number (the
identity, not a field), Rarity (kept as rarity), CardType (the type),
Description (the text), Flavor Text, BANDAI Disclaimer, GTIN.

Only cards in a deck at the start of a game are cards here (user,
2026-11-07): tokens a card makes mid-game, and the pieces every player
gets at setup whatever their deck (Gundam's EX Base and EX Resource,
Dragon Ball's Energy Markers), are not. Resource-like cards that start
in a deck (Gundam's Resources) are.

Profiled on the 2026-11-06 snapshot (scratch scans, not kept).
"""

import re

from src.data_refinement.card_binder.tcgcsv.catalog import TCGCSV_FORMAT
from src.data_refinement.card_binder.tcgjson.game_spec import (
    LEADING_CARD_NUMBER,
    NAME_ENDING_IN_CARD_NUMBER,
    CatalogField,
    FieldKind,
    TcgjsonGameSpec,
)
from src.schema.game_id import GameId

# Gundam: the card number is the card ("GD01-001"), except a Resource:
# every Resource has the same text and its own number per artwork
# ("R-001", promo "RP-004"), so a Resource is its name, one card. A
# name-keyed group registers no PRINTED_NUMBER aliases, so a Gundam deck
# source naming Resources by number ("R-001") must match them by name.
# Attack and Hit Points are text: a Pilot prints its bonus, "+1". Tokens
# ("T-030", a Unit a card creates) are not deck cards; nor are EX Base
# and EX Resource (placed at setup).
GUNDAM_SPEC = TcgjsonGameSpec(
    slug="gundam",
    game=GameId.GUNDAM,
    fields=(
        CatalogField("Color", "color", FieldKind.LIST),
        CatalogField("Level", "level", FieldKind.INT),
        CatalogField("Cost", "cost", FieldKind.INT),
        CatalogField("Attack Points", "ap", FieldKind.TEXT),
        CatalogField("Hit Points", "hp", FieldKind.TEXT),
        CatalogField("Trait", "traits", FieldKind.TEXT),
        CatalogField("Link Condition", "link_condition", FieldKind.TEXT),
        CatalogField("Zone", "zones", FieldKind.LIST),
    ),
    rarity_ladder=(
        "Common", "C+", "C++", "Uncommon", "U+", "Rare", "R+",
        "Legend Rare", "LR+", "LR++",
    ),  # fmt: skip
    excluded_types=("EX Base", "EX Resource"),
    card_number=re.compile(r"^(?!RP?-)\S+"),
    excluded_numbers=re.compile(r"^T-"),
    catalog_format=TCGCSV_FORMAT,
)

# Dragon Ball Super Fusion World: names may repeat the number ("Piccolo -
# FB10-068"); Cost carries its color ("2(Y)") and a Leader's Power its
# awakened side ("15000/20000"), so both are text. Energy Markers have no
# CardType and are dropped as untyped.
DRAGON_BALL_FUSION_WORLD_SPEC = TcgjsonGameSpec(
    slug="dragon-ball-fusion-world",
    game=GameId.DRAGON_BALL_FUSION_WORLD,
    fields=(
        CatalogField("Color", "color", FieldKind.LIST),
        CatalogField("Cost", "cost", FieldKind.TEXT),
        CatalogField("Power", "power", FieldKind.TEXT),
        CatalogField("Combo Power", "combo_power", FieldKind.INT),
        CatalogField("Character Traits", "traits", FieldKind.LIST),
    ),
    rarity_ladder=(
        "Common", "Uncommon", "Rare", "Super Rare", "Leader", "Secret Rare",
    ),  # fmt: skip
    card_number=LEADING_CARD_NUMBER,
    name_suffix=NAME_ENDING_IN_CARD_NUMBER,
    catalog_format=TCGCSV_FORMAT,
)

# Dragon Ball Super Masters: some real cards have no CardType on any
# printing (the Leader "Vegito // Going All In, SSB Vegito", BT3-055), so
# untyped numbered products are cards, minus Energy Markers ("M-45",
# "BT31-EM03") and merit cards (numbered "1", "2", ...); Token cards
# ("Ghost Token", made by a card mid-game) are not deck cards either. A pre-release
# printing's number ends "_PR" ("BT31-055_PR") and is the same card.
# Era (the anime saga) is flavor, not game text.
DRAGON_BALL_MASTERS_SPEC = TcgjsonGameSpec(
    slug="dragon-ball-masters",
    game=GameId.DRAGON_BALL_MASTERS,
    fields=(
        CatalogField("Color", "color", FieldKind.LIST),
        CatalogField("EnergyColor Cost", "energy_cost", FieldKind.TEXT),
        CatalogField("Z Energy Cost", "z_energy_cost", FieldKind.INT),
        CatalogField("Power", "power", FieldKind.TEXT),
        CatalogField("Combo Power", "combo_power", FieldKind.INT),
        CatalogField("Combo Energy", "combo_energy", FieldKind.INT),
        CatalogField("Character", "character", FieldKind.TEXT),
        CatalogField("SpecialTrait", "special_traits", FieldKind.TEXT),
    ),
    rarity_ladder=(
        "Common", "Uncommon", "Rare", "Starter Rare", "Super Rare",
        "Expansion Rare", "Special Rare", "Secret Rare",
    ),  # fmt: skip
    excluded_types=("Token",),
    card_number=re.compile(r"^[^\s_]+"),
    excluded_numbers=re.compile(r"^M-|-EM\d|^\d+$"),
    untyped_cards=True,
    catalog_format=TCGCSV_FORMAT,
)

# Battle Spirits Saga: costs and BP per level (LV I-III); Reduction
# Symbols is text (a count, "2", or colors, "PPP").
BATTLE_SPIRITS_SAGA_SPEC = TcgjsonGameSpec(
    slug="battle-spirits-saga",
    game=GameId.BATTLE_SPIRITS_SAGA,
    fields=(
        CatalogField("Color", "color", FieldKind.LIST),
        CatalogField("Cost", "cost", FieldKind.INT),
        CatalogField("Reduction Symbols", "reduction_symbols", FieldKind.TEXT),
        CatalogField("Symbols", "symbols", FieldKind.INT),
        CatalogField("Spirit Type", "family", FieldKind.TEXT),
        CatalogField("LV I Cost", "lv1_core", FieldKind.INT),
        CatalogField("LV I BP", "lv1_bp", FieldKind.INT),
        CatalogField("LV II Cost", "lv2_core", FieldKind.INT),
        CatalogField("LV II BP", "lv2_bp", FieldKind.INT),
        CatalogField("LV III Cost", "lv3_core", FieldKind.INT),
        CatalogField("LV III BP", "lv3_bp", FieldKind.INT),
    ),
    rarity_ladder=(
        "Common", "Uncommon", "Rare", "X Rare", "Special Rare",
    ),  # fmt: skip
    card_number=LEADING_CARD_NUMBER,
    catalog_format=TCGCSV_FORMAT,
)


# Every game this project takes from tcgcsv
ALL_SPECS = (
    GUNDAM_SPEC,
    DRAGON_BALL_FUSION_WORLD_SPEC,
    DRAGON_BALL_MASTERS_SPEC,
    BATTLE_SPIRITS_SAGA_SPEC,
)
