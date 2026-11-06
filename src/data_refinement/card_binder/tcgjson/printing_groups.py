"""From a catalog's products (one per printing) to cards: which products
are cards, which printings are one card, and each card's lean content.

Identity is the base name: the product name without its trailing
printing tags, "(...)" or "[...]" ("Elsa - Snow Queen (Epic)" -> "Elsa -
Snow Queen"), and without the spec's name_suffix; compared ignoring case
and whether a subtitle follows ", " or " - " (Riftbound writes both).
For a game whose spec sets card_number, it is the card number instead
(many One Piece cards share a name); a printing without one falls back
to its name. A card's name is the most common base name of its
printings.

Identity is not content, as in hearthstonejson/: TCGplayer types each
product's text by hand, so printings of one card disagree by spacing,
symbol rendering ("4 *" vs "4 Ink") and the odd wrong field (Lorcana,
weekly-20260831: 2,826 distinct contents for 2,548 names). Each field is
instead the majority of the printings' normalized values, so one
mistyped printing is outvoted. An errata'd card stays one card: when a
group has printings tagged "(Errata Version)", only they vote, so the
current text wins. A tie goes to the newest printing (highest
product_id), which carries the latest wording.

rarity and set are the representative printing's (card_binder/README.md
rule 10): its lowest rarity on the spec's ladder, then the earliest set
release (a set without a date last), then the lowest product_id.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from src.data_refinement.card_binder.lean_content import order_keys, strip_noise
from src.data_refinement.card_binder.tcgjson.catalog import (
    TcgjsonCatalog,
    TcgjsonProduct,
)
from src.data_refinement.card_binder.tcgjson.game_spec import (
    FieldKind,
    TcgjsonGameSpec,
)
from src.data_refinement.card_binder.tcgjson.rules_text import clean_rules_text
from src.schema.json_types import JsonObject, JsonValue

# One or more trailing "(...)" or "[...]" groups: TCGplayer's printing tags
_PRINTING_TAGS = re.compile(r"(\s*(\([^()]*\)|\[[^\[\]]*\]))+\s*$")
# A subtitle separator, ", " or " - ", for the name key
_SUBTITLE_SEPARATOR = re.compile(r"\s*(,|\s-)\s+")
_WHITESPACE_RUN = re.compile(r"\s+")
# The customAttributes key of a printing's card number
_NUMBER_ATTRIBUTE = "number"
# _identity_key's two kinds of key
_NUMBER_KEY, _NAME_KEY_KIND = "number", "name"
_ERRATA_TAG = "(Errata Version)"
# Sorts after every real ISO date
_NO_RELEASE_DATE = "￿"

# raw_content keys outside the spec's fields
_NAME_KEY, _TYPE_KEY = "name", "type"
_RARITY_KEY, _SET_KEY, _TEXT_KEY = "rarity", "set", "text"


@dataclass(frozen=True)
class PrintingGroup:
    """One card and every product that prints it.

    name: the card's name, the most common base name of its printings
        (a tie goes to the newest).
    products: its printings, lowest product_id first.
    card_numbers: the numbers a deck list may name the card by, for a
        card that is its card number (spec.card_number): that number
        first, then each printing's full printed number where it differs
        (Union Arena: "BLC-1-041", then "UE01BT/BLC-1-041",
        "UEPR/BLC-1-041"); empty for a card that is its name.
    """

    name: str
    products: tuple[TcgjsonProduct, ...]
    card_numbers: tuple[str, ...] = ()


def group_printings(
    catalog: TcgjsonCatalog, spec: TcgjsonGameSpec
) -> list[PrintingGroup]:
    """Group the catalog's card products into one PrintingGroup per card
    (its base name, or card number; see the module docstring).

    A product is a card if it has a type (card_types, or
    spec.type_attribute) none of which is in spec.excluded_types, and its
    set's name starts with none of spec.excluded_set_prefixes.

    Inputs: catalog, spec.
    Output: groups in order of their lowest product_id; each group's
        products lowest product_id first.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> [g.name for g in group_printings(catalog, LORCANA_SPEC)][:1]
        ['Mickey Mouse - Brave Little Tailor']
    """
    result: list[PrintingGroup] = []
    groups: dict[tuple[str, str], list[TcgjsonProduct]] = {}
    for product in sorted(catalog.products, key=lambda p: p.product_id):
        if not _is_card(product, catalog, spec):
            continue
        groups.setdefault(_identity_key(product, spec), []).append(product)

    # Name each group by its printings' vote, newest first
    for (kind, key), products in groups.items():
        names = [base_name(p.name, spec.name_suffix) for p in reversed(products)]
        numbers = _card_numbers(key, products) if kind == _NUMBER_KEY else ()
        result.append(PrintingGroup(str(_vote(names)), tuple(products), numbers))
    return result


def representative_printing(
    group: PrintingGroup, catalog: TcgjsonCatalog, spec: TcgjsonGameSpec
) -> TcgjsonProduct:
    """The printing whose rarity and set the card keeps: lowest rank on
    spec.rarity_ladder (off-ladder rarities last), then earliest set
    release_date (none last), then lowest product_id.

    Inputs: group (non-empty), catalog (for set release dates), spec.
    Output: one of group.products.
    Side effects: none.
    Exceptions: ValueError if group has no products.

    Example:
        >>> representative_printing(mickey, catalog, LORCANA_SPEC).rarity
        'Legendary'
    """
    if not group.products:
        raise ValueError(f"printing group {group.name!r} has no products")
    return min(group.products, key=lambda p: _representative_rank(p, catalog, spec))


def build_card_content(
    group: PrintingGroup, catalog: TcgjsonCatalog, spec: TcgjsonGameSpec
) -> JsonObject:
    """The card's lean raw_content: name, type, each spec field, rarity,
    set and text, each field the vote of the voting printings (see the
    module docstring), then strip_noise and order_keys (name and type
    first, text last). A FLAG field's False survives strip_noise.

    Inputs: group (non-empty), catalog, spec.
    Output: a new JsonObject.
    Side effects: none.
    Exceptions: ValueError if group has no products.

    Example:
        >>> build_card_content(mickey, catalog, LORCANA_SPEC)
        {'name': 'Mickey Mouse - Brave Little Tailor', 'type': 'Character',
         'ink': 'Ruby', 'cost': 8, 'inkwell': True, 'strength': 5,
         'willpower': 5, 'lore': 4, 'classifications': ['Dreamborn', 'Hero'],
         'rarity': 'Legendary', 'set': 'The First Chapter',
         'text': 'Evasive (Only characters with Evasive can challenge this character.)'}
    """
    representative = representative_printing(group, catalog, spec)
    voters = _voting_printings(group)

    # Vote each field across the voting printings
    content: JsonObject = {_NAME_KEY: group.name}
    content[_TYPE_KEY] = _vote([_card_type(p, spec) for p in voters])
    flags: JsonObject = {}
    for field in spec.fields:
        values = [field.normalize(p.attributes.get(field.attribute)) for p in voters]
        target = flags if field.kind is FieldKind.FLAG else content
        target[field.card_key] = _vote(values)
    content[_RARITY_KEY] = representative.rarity
    content[_SET_KEY] = catalog.sets[representative.set_id].name
    content[_TEXT_KEY] = _vote([_clean_text_or_none(p.rules_text) for p in voters])

    # Lean it: empties out (a FLAG's False kept), then the key order
    lean = strip_noise(content)
    lean.update({key: value for key, value in flags.items() if value is not None})
    leading = (
        _NAME_KEY,
        _TYPE_KEY,
        *(field.card_key for field in spec.fields),
        _RARITY_KEY,
        _SET_KEY,
    )
    return order_keys(lean, leading, (_TEXT_KEY,))


def base_name(name: str, suffix: re.Pattern[str] | None = None) -> str:
    """name without its trailing printing tags ("(...)", "[...]"), then
    without suffix (a spec's name_suffix) if given; whitespace runs
    collapsed to one space.

    Inputs: name, suffix. Output: str (name itself, stripped, if nothing
        would be left).
    Side effects: none. Exceptions: none.

    Example:
        >>> base_name("Elsa - Snow Queen (Epic) (Foil)")
        'Elsa - Snow Queen'
        >>> base_name("Usopp (3-on-3 Cup) [Winner]")
        'Usopp'
    """
    stripped = _PRINTING_TAGS.sub("", name).strip()
    if suffix is not None:
        stripped = suffix.sub("", stripped).strip()
    return _WHITESPACE_RUN.sub(" ", stripped or name.strip())


def _identity_key(product: TcgjsonProduct, spec: TcgjsonGameSpec) -> tuple[str, str]:
    """Which card product prints: ("number", its card number) if
    spec.card_number matches its number attribute, else ("name", its base
    name casefolded, ", " read as " - ").
    Inputs: product, spec. Output: the key.
    Side effects: none. Exceptions: none."""
    number = product.attributes.get(_NUMBER_ATTRIBUTE)
    if spec.card_number is not None and isinstance(number, str):
        match = spec.card_number.search(number)
        if match:
            return _NUMBER_KEY, match.group(0)
    name = base_name(product.name, spec.name_suffix).casefold()
    return _NAME_KEY_KIND, _SUBTITLE_SEPARATOR.sub(" - ", name)


def _card_numbers(number: str, products: list[TcgjsonProduct]) -> tuple[str, ...]:
    """number, then each product's full printed number (its number
    attribute's first token) not yet listed, lowest product_id first.
    Inputs: number (the group's identity number), products.
    Output: tuple of distinct numbers, number first.
    Side effects: none. Exceptions: none."""
    result = [number]
    for product in products:
        printed = product.attributes.get(_NUMBER_ATTRIBUTE)
        tokens = printed.split() if isinstance(printed, str) else []
        if tokens and tokens[0] not in result:
            result.append(tokens[0])
    return tuple(result)


def _is_card(
    product: TcgjsonProduct, catalog: TcgjsonCatalog, spec: TcgjsonGameSpec
) -> bool:
    """Whether product is a deck card (see group_printings).
    Inputs: product, catalog, spec. Output: bool.
    Side effects: none. Exceptions: none."""
    card_type = _card_type(product, spec)
    if card_type is None:
        return False
    types = card_type if isinstance(card_type, list) else [card_type]
    if any(t in spec.excluded_types for t in types):
        return False
    return not catalog.sets[product.set_id].name.startswith(spec.excluded_set_prefixes)


def _voting_printings(group: PrintingGroup) -> tuple[TcgjsonProduct, ...]:
    """The printings whose values count: the errata-tagged ones if the
    group has any, else all. Newest first, so a tie goes to the newest.
    Inputs: group. Output: tuple of products (non-empty if group is).
    Side effects: none. Exceptions: none."""
    newest_first = tuple(reversed(group.products))
    errata = tuple(p for p in newest_first if _ERRATA_TAG in p.name)
    return errata or newest_first


def _vote(values: Sequence[JsonValue]) -> JsonValue:
    """The most common non-None value; a tie goes to the earliest in
    values (the caller passes newest first). None if every value is None.
    Lists compare by value.
    Inputs: values. Output: one of them, or None.
    Side effects: none. Exceptions: none."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    # Counter keeps first-seen order among equal counts, so a tie goes first
    counts = Counter(json.dumps(value, sort_keys=True) for value in present)
    winner = counts.most_common(1)[0][0]
    return next(v for v in present if json.dumps(v, sort_keys=True) == winner)


def _representative_rank(
    product: TcgjsonProduct, catalog: TcgjsonCatalog, spec: TcgjsonGameSpec
) -> tuple[int, str, int]:
    """Sort key for representative_printing: (ladder rank, off-ladder
    after it; set release_date, none last; product_id).
    Inputs: product, catalog, spec. Output: the key.
    Side effects: none. Exceptions: none."""
    ladder = spec.rarity_ladder
    rank = ladder.index(product.rarity) if product.rarity in ladder else len(ladder)
    release_date = catalog.sets[product.set_id].release_date or _NO_RELEASE_DATE
    return rank, release_date, product.product_id


def _card_type(product: TcgjsonProduct, spec: TcgjsonGameSpec) -> JsonValue:
    """The product's type: its spec.type_attribute value if the spec
    names one, else its card_types; each cleaned as a TEXT value (so
    "None" is no type); a string if one type, a list if several, None
    if none.
    Inputs: product, spec. Output: str | list | None.
    Side effects: none. Exceptions: none."""
    raw: tuple[JsonValue, ...] = product.card_types
    if spec.type_attribute is not None:
        raw = (product.attributes.get(spec.type_attribute),)
    types: list[JsonValue] = []
    for value in raw:
        cleaned = FieldKind.TEXT.normalize(value)
        if cleaned is not None:
            types.append(cleaned)
    if not types:
        return None
    if len(types) == 1:
        return types[0]
    return types


def _clean_text_or_none(rules_text: str | None) -> str | None:
    """clean_rules_text, with None and an all-markup text as None.
    Inputs: rules_text. Output: str | None.
    Side effects: none. Exceptions: none."""
    if rules_text is None:
        return None
    return clean_rules_text(rules_text) or None
