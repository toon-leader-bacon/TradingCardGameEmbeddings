"""Reading one tcgcsv catalog file into the TcgjsonCatalog the tcgjson
pipeline runs on: the parse boundary of card_binder/tcgcsv/.

A catalog (<slug>.json.gz, landed by
src/data_retrieval/tcgcsv/downloader.py) is {category_id, groups[],
products[]}, TCGplayer's own field names. The mapping:

    group                      -> TcgjsonSet
      groupId, name, publishedOn  set_id, name, release_date
    product                    -> TcgjsonProduct
      productId, name, groupId    product_id, name, set_id
      extendedData[]              attributes, {name: value}; its "Number"
                                  stored under catalog.NUMBER_ATTRIBUTE
      extendedData Rarity         rarity
      extendedData CardType       card_types (one type, or none)
      extendedData Description    rules_text (raw HTML)

Image URLs, the store URL, cleanName, presale info and modifiedOn are
not read. A sealed product (a booster box) has no extendedData Number or
CardType; group_printings drops it as an untyped product.
"""

from pathlib import Path
from types import MappingProxyType

from src.data_refinement.card_binder.tcgjson.catalog import (
    NUMBER_ATTRIBUTE,
    CatalogFormat,
    TcgjsonCatalog,
    TcgjsonProduct,
    TcgjsonSet,
    assemble_catalog,
    load_catalog_lists,
)
from src.data_retrieval.tcgcsv.downloader import CATALOG_FILE_SUFFIX
from src.schema.data_source import DataSource
from src.schema.json_checks import require_int, require_nonempty_str
from src.schema.json_types import JsonValue

# extendedData names this reader maps onto TcgjsonProduct fields
_NUMBER, _RARITY, _CARD_TYPE, _DESCRIPTION = (
    "Number",
    "Rarity",
    "CardType",
    "Description",
)


def read_tcgcsv_catalog(path: Path) -> TcgjsonCatalog:
    """Read one tcgcsv catalog file (.json.gz, or plain .json) into a
    TcgjsonCatalog.

    Inputs: path (a <slug>.json.gz).
    Output: TcgjsonCatalog with every group as a set and every product,
        in file order.
    Side effects: reads path.
    Exceptions: FileNotFoundError if path is missing; ValueError if it
        is not JSON or lacks "groups"/"products" lists; KeyError/
        TypeError if a group or product lacks an int id or a string
        name, or a product names a group the catalog lacks.

    Example:
        >>> catalog = read_tcgcsv_catalog(
        ...     Path("data/raw/tcgcsv/20261106T200528Z/gundam.json.gz"))
        >>> len(catalog.products)
        2332
    """
    raw_groups, raw_products = load_catalog_lists(path, "groups", "products")
    sets = [_parse_group(raw) for raw in raw_groups]
    products = [_parse_product(raw) for raw in raw_products]
    return assemble_catalog(sets, products)


def _parse_group(raw: JsonValue) -> TcgjsonSet:
    """One groups[] entry as a set; publishedOn ("2025-07-25T00:00:00")
    is its release date, None if missing.
    Inputs: raw. Output: TcgjsonSet. Side effects: none.
    Exceptions: KeyError/TypeError if groupId or name is missing or of
        the wrong type."""
    if not isinstance(raw, dict):
        raise TypeError(f"tcgcsv group is not an object: {raw!r}")
    published_on = raw.get("publishedOn")
    return TcgjsonSet(
        set_id=require_int(raw["groupId"]),
        name=require_nonempty_str(raw["name"]),
        release_date=published_on if isinstance(published_on, str) else None,
    )


def _parse_product(raw: JsonValue) -> TcgjsonProduct:
    """One products[] entry (see the module docstring's mapping); a
    missing Rarity reads as "", a missing CardType as no type, a missing
    Description as None.
    Inputs: raw. Output: TcgjsonProduct. Side effects: none.
    Exceptions: KeyError/TypeError if productId, name or groupId is
        missing or of the wrong type."""
    if not isinstance(raw, dict):
        raise TypeError(f"tcgcsv product is not an object: {raw!r}")
    attributes = _extended_data(raw.get("extendedData"))
    rarity, card_type = attributes.get(_RARITY), attributes.get(_CARD_TYPE)
    description = attributes.get(_DESCRIPTION)
    return TcgjsonProduct(
        product_id=require_int(raw["productId"]),
        name=require_nonempty_str(raw["name"]),
        set_id=require_int(raw["groupId"]),
        rarity=rarity if isinstance(rarity, str) else "",
        card_types=(card_type,) if isinstance(card_type, str) and card_type else (),
        rules_text=description if isinstance(description, str) else None,
        attributes=MappingProxyType(attributes),
    )


def _extended_data(raw: JsonValue) -> dict[str, JsonValue]:
    """A product's extendedData list as {name: value}, "Number" stored
    under NUMBER_ATTRIBUTE; entries without a string name are skipped,
    a repeated name keeps its last value.
    Inputs: raw (the extendedData value; anything but a list reads as
        empty). Output: dict. Side effects: none. Exceptions: none."""
    result: dict[str, JsonValue] = {}
    entries = raw if isinstance(raw, list) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        result[NUMBER_ATTRIBUTE if name == _NUMBER else name] = entry.get("value")
    return result


# tcgcsv's daily TCGplayer mirror (src/data_retrieval/tcgcsv/)
TCGCSV_FORMAT = CatalogFormat(
    DataSource.TCGCSV, CATALOG_FILE_SUFFIX, read_tcgcsv_catalog
)
