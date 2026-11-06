"""What differs between tcgjson games, as data: which customAttributes
become which raw_content keys, of what kind; the rarity ladder; which sets
and card types are not deck cards; where the type is read from; whether a
card is its name or its card number; and what printing detail a name
carries. One TcgjsonGameSpec constant per game, here.

Every tcgjson game shares the catalog layout (catalog.py); only the
customAttributes keys differ (checked across all nine games' schema
profiles, release weekly-20260831). A game whose fields need a transform
no FieldKind expresses gets a new FieldKind (and its case in
FieldKind.normalize), not per-game code.
"""

import re
from dataclasses import dataclass
from enum import Enum

from src.data_refinement.card_binder.tcgjson.rules_text import clean_rules_text
from src.schema.game_id import GameId
from src.schema.json_types import JsonValue

_WHITESPACE_RUN = re.compile(r"\s+")
_WHOLE_NUMBER = re.compile(r"-?\d+")
_LIST_SEPARATOR = ";"
_FLAG_VALUES = {"Yes": True, "No": False}
# TCGplayer's placeholder for an empty value (Riftbound's colorless domain)
_MISSING = "None"
# Card-number rules shared by specs: the number's first token ("BT5-086 SR"
# -> "BT5-086"), the part after its "/" ("UE01BT/BLC-1-041" -> "BLC-1-041"),
# and a name that repeats the number ("Upamon - BT1-003")
LEADING_CARD_NUMBER = re.compile(r"^\S+")
CARD_NUMBER_AFTER_SLASH = re.compile(r"(?<=/)[^/\s]+")
NAME_ENDING_IN_CARD_NUMBER = re.compile(r"\s+-\s+[A-Z]+\d*-\d+$")


class FieldKind(Enum):
    """How one customAttributes value becomes a raw_content value
    (FieldKind.normalize).

    TEXT: a string, whitespace collapsed.
    INT: "8" -> 8; a value that is not a whole number is dropped.
    LIST: "Dreamborn;Hero" -> ["Dreamborn", "Hero"]; a JSON list of
        strings is kept, cleaned the same way.
    FLAG: "Yes" -> True, "No" -> False (False is kept: it is
        information); any other value ("Unknown") is dropped.
    RULES: rules-text HTML, cleaned as the card's text is
        (clean_rules_text): Digimon's inherited effect.
    The placeholder "None" is a missing value, for every kind.
    """

    TEXT = "text"
    INT = "int"
    LIST = "list"
    FLAG = "flag"
    RULES = "rules"

    def normalize(self, value: JsonValue) -> JsonValue:
        """value as this kind says (see the class docstring); None for a
        missing or unusable value.

        Inputs: value (as the catalog holds it, any JSON value).
        Output: str, int, bool, list[str], or None.
        Side effects: none. Exceptions: none.

        Example:
            >>> FieldKind.LIST.normalize("Dreamborn;Hero")
            ['Dreamborn', 'Hero']
        """
        if self is FieldKind.LIST:
            return _normalize_list(value)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return None
        if self is FieldKind.RULES:
            text = clean_rules_text(str(value))
        else:
            text = _WHITESPACE_RUN.sub(" ", str(value)).strip()
        if text == _MISSING:
            return None
        if self is FieldKind.INT:
            return int(text) if _WHOLE_NUMBER.fullmatch(text) else None
        if self is FieldKind.FLAG:
            return _FLAG_VALUES.get(text)
        return text or None


@dataclass(frozen=True)
class CatalogField:
    """One kept field.

    attribute: the customAttributes key ("costInk").
    card_key: the raw_content key, in the game's own terms ("cost").
    kind: how the value is normalized.
    """

    attribute: str
    card_key: str
    kind: FieldKind

    def normalize(self, value: JsonValue) -> JsonValue:
        """value as self.kind says (FieldKind.normalize).

        Inputs: value. Output: str, int, bool, list[str], or None.
        Side effects: none. Exceptions: none.

        Example:
            >>> CatalogField("c", "c", FieldKind.LIST).normalize("Dreamborn;Hero")
            ['Dreamborn', 'Hero']
        """
        return self.kind.normalize(value)


@dataclass(frozen=True)
class TcgjsonGameSpec:
    """One game's reading of its tcgjson catalog.

    slug: tcgjson's name for the game, the catalog file's stem.
    game: the GameId its cards are stored under.
    fields: the kept fields, in raw_content order (after name and type;
        rarity, set and text follow them).
    rarity_ladder: the game's printed rarities, least to most
        restrictive; a rarity not on it (Promo) ranks after all of them.
    excluded_set_prefixes: a product whose set name starts with one of
        these is not a deck card (Lorcana's co-op Illumineer's Quest).
    excluded_types: a product of one of these types is not a deck card
        (One Piece's DON!! and Union Arena's Action Point cards: kept
        outside the deck, no text, one product per artwork).
    type_attribute: the customAttributes key holding the card's type,
        for a game whose metadata.cardTypes is coarse or often missing
        (Yu-Gi-Oh's cardTypeB, "Effect Monster"); None: cardTypes.
    card_number: for a game where many cards share a name (One Piece
        has dozens of "Monkey.D.Luffy"), a card is its card number: the
        first match of this pattern in the product's "number" attribute
        (LEADING_CARD_NUMBER: "OP01-024"; CARD_NUMBER_AFTER_SLASH:
        "UE01BT/BLC-1-041" -> "BLC-1-041"). None, or a product with no match: its name.
    name_suffix: printing detail stripped from the end of a name after
        its parenthetical tags (Star Wars: Unlimited's token backs,
        "Capital City // Shield"); None: nothing more.
    """

    slug: str
    game: GameId
    fields: tuple[CatalogField, ...]
    rarity_ladder: tuple[str, ...]
    excluded_set_prefixes: tuple[str, ...] = ()
    excluded_types: tuple[str, ...] = ()
    type_attribute: str | None = None
    card_number: re.Pattern[str] | None = None
    name_suffix: re.Pattern[str] | None = None


def _normalize_list(value: JsonValue) -> JsonValue:
    """A LIST value: split on ";" if a string, kept if a list; items
    whitespace-collapsed, empty and non-string items dropped.
    Inputs: value. Output: a list of strings, or None if nothing is left.
    Side effects: none. Exceptions: none."""
    if isinstance(value, str):
        items: list[JsonValue] = list(value.split(_LIST_SEPARATOR))
    elif isinstance(value, list):
        items = value
    else:
        return None
    result: list[JsonValue] = []
    for item in items:
        if isinstance(item, str) and item.strip() and item.strip() != _MISSING:
            result.append(_WHITESPACE_RUN.sub(" ", item).strip())
    return result or None


# Lorcana, profiled on release weekly-20260831. Dropped attributes:
# characterVersion (already in the name), property (the Disney franchise,
# a store category, not game text), flavorText, and the store-wide noise
# (description = rulesText, releaseDate, number, rarityDbName, detailNote,
# promoType).
LORCANA_SPEC = TcgjsonGameSpec(
    slug="lorcana",
    game=GameId.LORCANA,
    fields=(
        CatalogField("inkType", "ink", FieldKind.TEXT),
        CatalogField("costInk", "cost", FieldKind.INT),
        CatalogField("inkwellIcononCard", "inkwell", FieldKind.FLAG),
        CatalogField("strength", "strength", FieldKind.INT),
        CatalogField("willpower", "willpower", FieldKind.INT),
        CatalogField("loreValue", "lore", FieldKind.INT),
        CatalogField("moveCost", "move_cost", FieldKind.INT),
        CatalogField("classification", "classifications", FieldKind.LIST),
    ),
    rarity_ladder=(
        "Common", "Uncommon", "Rare", "Super Rare", "Legendary", "Epic",
        "Enchanted", "Iconic",
    ),  # fmt: skip
    excluded_set_prefixes=("Illumineer's Quest",),
)


# The games below were profiled on release weekly-20260831. Every game
# drops the store-wide noise: description (= rulesText, or the text
# itself where rulesText is missing; catalog.py reads it), cardType
# (= cardTypes), number, releaseDate, rarityDbName, detailNote,
# disclaimer, flavorText.

# Yu-Gi-Oh: cardTypes is only Main/Extra Deck Monster, Spell or Trap, and
# missing on most Pendulum printings; cardTypeB ("Xyz/Effect Monster",
# "Quick-Play Spell") is the card's type. The "Yu-Gi-Oh! Tokens" set holds
# tokens, not deck cards.
YUGIOH_SPEC = TcgjsonGameSpec(
    slug="yugioh",
    game=GameId.YUGIOH,
    fields=(
        CatalogField("attribute", "attribute", FieldKind.LIST),
        CatalogField("monsterType", "monster_type", FieldKind.LIST),
        CatalogField("level", "level", FieldKind.INT),
        CatalogField("linkRating", "link_rating", FieldKind.INT),
        CatalogField("linkArrows", "link_arrows", FieldKind.LIST),
        CatalogField("attack", "atk", FieldKind.INT),
        CatalogField("defense", "def", FieldKind.INT),
    ),
    rarity_ladder=(
        "Common / Short Print", "Rare", "Super Rare", "Ultra Rare",
        "Secret Rare", "Ultimate Rare",
    ),  # fmt: skip
    excluded_set_prefixes=("Yu-Gi-Oh! Tokens",),
    type_attribute="cardTypeB",
)

# Digimon: the card number is the card (dozens of distinct "Agumon"); the
# number carries a rarity suffix ("BT5-086 SR"), and a few names repeat
# it ("Upamon - BT1-003"). Its text lives only in description. origins
# (the promo's event) is store detail.
DIGIMON_SPEC = TcgjsonGameSpec(
    slug="digimon-card-game",
    game=GameId.DIGIMON,
    fields=(
        CatalogField("color", "color", FieldKind.LIST),
        CatalogField("playCost", "play_cost", FieldKind.INT),
        CatalogField("levelLv", "level", FieldKind.INT),
        CatalogField("digimonForm", "form", FieldKind.TEXT),
        CatalogField("digimonAttribute", "attribute", FieldKind.TEXT),
        CatalogField("digimonType", "digimon_type", FieldKind.TEXT),
        CatalogField("digimonPowerDp", "dp", FieldKind.INT),
        CatalogField("digivolve1Color", "digivolve_1_color", FieldKind.TEXT),
        CatalogField("digivolve1Level", "digivolve_1_level", FieldKind.TEXT),
        CatalogField("digivolve1Cost", "digivolve_1_cost", FieldKind.INT),
        CatalogField("digivolve2Color", "digivolve_2_color", FieldKind.TEXT),
        CatalogField("digivolve2Level", "digivolve_2_level", FieldKind.TEXT),
        CatalogField("digivolve2Cost", "digivolve_2_cost", FieldKind.INT),
        CatalogField("digivolve3Level", "digivolve_3_level", FieldKind.TEXT),
        CatalogField("inheritedEffect", "inherited_effect", FieldKind.RULES),
        CatalogField("securityEffect", "security_effect", FieldKind.RULES),
    ),
    rarity_ladder=(
        "Common", "Uncommon", "Rare", "Super Rare", "Secret Rare",
        "Special Rare", "Ultimate Rare",
    ),  # fmt: skip
    card_number=LEADING_CARD_NUMBER,
    name_suffix=NAME_ENDING_IN_CARD_NUMBER,
)

# One Piece: the card number is the card, as in Digimon ("OP01-024",
# "P-001"); "subtypes" is what the game calls a card's Types ("Straw Hat
# Crew").
ONE_PIECE_SPEC = TcgjsonGameSpec(
    slug="one-piece",
    game=GameId.ONE_PIECE,
    fields=(
        CatalogField("color", "color", FieldKind.LIST),
        CatalogField("cost", "cost", FieldKind.INT),
        CatalogField("life", "life", FieldKind.INT),
        CatalogField("power", "power", FieldKind.INT),
        CatalogField("counter", "counter", FieldKind.INT),
        CatalogField("attribute", "attribute", FieldKind.LIST),
        CatalogField("subtypes", "types", FieldKind.LIST),
    ),
    rarity_ladder=(
        "Common", "Uncommon", "Rare", "Super Rare", "Leader", "Secret Rare",
        "Treasure Rare",
    ),  # fmt: skip
    excluded_types=("DON!!",),
    card_number=LEADING_CARD_NUMBER,
    name_suffix=NAME_ENDING_IN_CARD_NUMBER,
)

# Star Wars: Unlimited: names carry the subtitle ("Luke Skywalker -
# Faithful Friend"), so the name is the card; a base's token back is a
# printing detail. epicAction (a leader's deploy condition) is not in
# rulesText.
STAR_WARS_UNLIMITED_SPEC = TcgjsonGameSpec(
    slug="star-wars-unlimited",
    game=GameId.STAR_WARS_UNLIMITED,
    fields=(
        CatalogField("aspect", "aspects", FieldKind.LIST),
        CatalogField("cost", "cost", FieldKind.INT),
        CatalogField("power", "power", FieldKind.INT),
        CatalogField("hp", "hp", FieldKind.INT),
        CatalogField("arenaType", "arena", FieldKind.TEXT),
        CatalogField("traits", "traits", FieldKind.LIST),
        CatalogField("epicAction", "epic_action", FieldKind.RULES),
    ),
    rarity_ladder=("Common", "Uncommon", "Rare", "Legendary", "Special"),
    name_suffix=re.compile(r"\s+//.*$"),
)

# Union Arena: the card number is the card; its part after "/" is shared
# by a card's reprints ("UE01BT/BLC-1-041", "UEPR/BLC-1-041"). Every card
# belongs to one series (the anime), which deck building is bound to, so
# it is kept, unlike Lorcana's franchise. bp is text: "3000+" is printed.
UNION_ARENA_SPEC = TcgjsonGameSpec(
    slug="union-arena",
    game=GameId.UNION_ARENA,
    fields=(
        CatalogField("seriesName", "series", FieldKind.TEXT),
        CatalogField("activationEnergy", "energy_color", FieldKind.LIST),
        CatalogField("requiredEnergy", "energy_cost", FieldKind.INT),
        CatalogField("actionPointCost", "ap_cost", FieldKind.INT),
        CatalogField("battlePointBp", "bp", FieldKind.TEXT),
        CatalogField("generatedEnergy", "energy_generation", FieldKind.TEXT),
        CatalogField("affinities", "traits", FieldKind.LIST),
        CatalogField("trigger", "trigger", FieldKind.RULES),
    ),
    rarity_ladder=("C", "U", "R", "SR", "UR", "SP"),
    excluded_types=("Action Point",),
    card_number=CARD_NUMBER_AFTER_SLASH,
)

# Riftbound: the name is the card ("Jinx, Loose Cannon"; printings write
# the separator as ", " or " - ", which identity ignores).
RIFTBOUND_SPEC = TcgjsonGameSpec(
    slug="riftbound",
    game=GameId.RIFTBOUND,
    fields=(
        CatalogField("domain", "domains", FieldKind.LIST),
        CatalogField("energyCost", "energy", FieldKind.INT),
        CatalogField("powerCost", "power", FieldKind.INT),
        CatalogField("might", "might", FieldKind.INT),
        CatalogField("tag", "tags", FieldKind.LIST),
    ),
    rarity_ladder=("Common", "Uncommon", "Rare", "Epic", "Showcase"),
)


# Every game this project takes from tcgjson
ALL_SPECS = (
    LORCANA_SPEC,
    YUGIOH_SPEC,
    DIGIMON_SPEC,
    ONE_PIECE_SPEC,
    STAR_WARS_UNLIMITED_SPEC,
    UNION_ARENA_SPEC,
    RIFTBOUND_SPEC,
)
