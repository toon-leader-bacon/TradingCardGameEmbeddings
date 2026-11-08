"""Reading one tcgjson catalog file into typed sets and products: the parse
boundary of card_binder/tcgjson/.

A catalog (<slug>.full.json.gz, landed by
src/data_retrieval/tcgjson/downloader.py) is {meta, sets[], products[]}:
one product per TCGplayer printing. Only the fields a card can need are
read; image URLs, foilings, collector numbers and the meta block are not.
A product's customAttributes stay a read-only mapping
(TcgjsonProduct.attributes): their keys are game-specific, read through
each game's TcgjsonGameSpec, except two every game shares: description
(read here, as the rules text where rulesText is missing) and number
(read by printing_groups.py, for a game whose cards are their numbers).

TcgjsonCatalog is the TCGplayer catalog shape the whole tcgjson pipeline
runs on, whichever site republishes it: CatalogFormat names a source's
file suffix, reader and DataSource, and each TcgjsonGameSpec names its
format (TCGJSON_FORMAT here; card_binder/tcgcsv/ adds TCGCSV_FORMAT).
"""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from src.schema.data_source import DataSource
from src.schema.json_types import JsonValue
from src.schema.json_checks import require_int, require_nonempty_str

# The customAttributes copy of rulesText
_DESCRIPTION_ATTRIBUTE = "description"
# The attributes key of a printing's card number, for every source
NUMBER_ATTRIBUTE = "number"


@dataclass(frozen=True)
class TcgjsonSet:
    """One TCGplayer set (group) of a catalog.

    release_date: ISO date-time as the catalog gives it
        ("2023-08-18T00:00:00"), so dates of one catalog compare as
        strings; None if the catalog gives none.
    """

    set_id: int
    name: str
    release_date: str | None


@dataclass(frozen=True)
class TcgjsonProduct:
    """One TCGplayer product: one printing of a card, or a non-card.

    card_types: the product's types (empty for a non-card such as a
        puzzle insert or a booster box). rules_text: its rules text, raw
        HTML, or None. attributes: its game-specific fields by the
        source's own names, read-only; the card number, if any, under
        NUMBER_ATTRIBUTE.

    How each source fills them: tcgjson from metadata.cardTypes,
    metadata.rulesText (else the description attribute, the only copy in
    Digimon's catalog) and metadata.customAttributes (read_catalog);
    tcgcsv from its extendedData list (card_binder/tcgcsv/catalog.py).
    """

    product_id: int
    name: str
    set_id: int
    rarity: str
    card_types: tuple[str, ...]
    rules_text: str | None
    attributes: Mapping[str, JsonValue]


@dataclass(frozen=True)
class TcgjsonCatalog:
    """One game's catalog: its sets by set_id, and every product."""

    sets: Mapping[int, TcgjsonSet]
    products: tuple[TcgjsonProduct, ...]


def read_catalog(path: Path) -> TcgjsonCatalog:
    """Read one catalog file (.json.gz, or plain .json) into a
    TcgjsonCatalog.

    Inputs: path (a <slug>.full.json.gz).
    Output: TcgjsonCatalog with every set and product, in file order.
    Side effects: reads path.
    Exceptions: FileNotFoundError if path is missing; ValueError if it
        is not JSON or lacks "sets"/"products" lists; KeyError/TypeError
        if a set or product lacks an int id or a string name, or a
        product names a set the catalog lacks.

    Example:
        >>> catalog = read_catalog(
        ...     Path("data/raw/tcgjson/20260831T174333Z/lorcana.full.json.gz"))
        >>> len(catalog.products)
        3393
    """
    raw_sets, raw_products = load_catalog_lists(path, "sets", "products")
    sets = [_parse_set(raw) for raw in raw_sets]
    products = [_parse_product(raw) for raw in raw_products]
    return assemble_catalog(sets, products)


def load_catalog_lists(
    path: Path, sets_key: str, products_key: str
) -> tuple[list[JsonValue], list[JsonValue]]:
    """A catalog file's raw set and product lists: the JSON object at
    path (gunzipped if .gz), its sets_key and products_key lists.

    Inputs: path, sets_key, products_key (the source's names for them).
    Output: (raw sets, raw products), unparsed.
    Side effects: reads path.
    Exceptions: FileNotFoundError if path is missing; ValueError if it
        is not a JSON object or either key is not a list.

    Example:
        >>> raw_sets, raw_products = load_catalog_lists(path, "sets", "products")
    """
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    raw_sets, raw_products = payload.get(sets_key), payload.get(products_key)
    if not isinstance(raw_sets, list) or not isinstance(raw_products, list):
        raise ValueError(f"{path} lacks {sets_key}/{products_key} lists")
    return raw_sets, raw_products


def assemble_catalog(
    sets: list[TcgjsonSet], products: list[TcgjsonProduct]
) -> TcgjsonCatalog:
    """A TcgjsonCatalog of parsed sets and products, every product's
    set checked to be among sets.

    Inputs: sets, products (in file order).
    Output: TcgjsonCatalog (sets by set_id; a repeated set_id keeps the
        last).
    Side effects: none.
    Exceptions: KeyError if a product names a set not in sets.

    Example:
        >>> assemble_catalog([TcgjsonSet(1, "Set", None)], []).sets[1].name
        'Set'
    """
    by_id = {item.set_id: item for item in sets}
    for product in products:
        if product.set_id not in by_id:
            raise KeyError(
                f"product {product.product_id} names unknown set {product.set_id}"
            )
    return TcgjsonCatalog(MappingProxyType(by_id), tuple(products))


def load_json(path: Path) -> JsonValue:
    """path's JSON, gunzipped if its name ends in .gz.
    Inputs: path. Output: the JSON value. Side effects: reads path.
    Exceptions: FileNotFoundError; ValueError if not JSON (json's
        JSONDecodeError) or bad gzip (gzip.BadGzipFile is an OSError)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            return json.load(file)
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _parse_set(raw: JsonValue) -> TcgjsonSet:
    """One sets[] entry.
    Inputs: raw. Output: TcgjsonSet. Side effects: none.
    Exceptions: KeyError/TypeError if setId or name is missing or of the
        wrong type."""
    if not isinstance(raw, dict):
        raise TypeError(f"catalog set is not an object: {raw!r}")
    release_date = raw.get("releaseDate")
    return TcgjsonSet(
        set_id=require_int(raw["setId"]),
        name=require_nonempty_str(raw["name"]),
        release_date=release_date if isinstance(release_date, str) else None,
    )


def _parse_product(raw: JsonValue) -> TcgjsonProduct:
    """One products[] entry; a missing metadata, cardTypes or
    customAttributes reads as empty, a missing rarity as "", a missing
    rulesText as the description attribute.
    Inputs: raw. Output: TcgjsonProduct. Side effects: none.
    Exceptions: KeyError/TypeError if productId, name or setId is
        missing or of the wrong type."""
    if not isinstance(raw, dict):
        raise TypeError(f"catalog product is not an object: {raw!r}")
    metadata = raw.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    card_types = metadata.get("cardTypes")
    attributes = metadata.get("customAttributes")
    rules_text = metadata.get("rulesText")
    if not isinstance(rules_text, str) and isinstance(attributes, dict):
        rules_text = attributes.get(_DESCRIPTION_ATTRIBUTE)
    rarity = raw.get("rarity")
    return TcgjsonProduct(
        product_id=require_int(raw["productId"]),
        name=require_nonempty_str(raw["name"]),
        set_id=require_int(raw["setId"]),
        rarity=rarity if isinstance(rarity, str) else "",
        card_types=(
            tuple(item for item in card_types or () if isinstance(item, str) and item)
            if isinstance(card_types, list)
            else ()
        ),
        rules_text=rules_text if isinstance(rules_text, str) else None,
        attributes=MappingProxyType(
            dict(attributes) if isinstance(attributes, dict) else {}
        ),
    )


@dataclass(frozen=True)
class CatalogFormat:
    """Where one source's catalogs come from and how they are read.

    data_source: the DataSource its product-id aliases and provenance
        are registered under (a TCGplayer productId either way).
    file_suffix: a landed catalog's file name is <slug><file_suffix>.
    read: reads one catalog file into a TcgjsonCatalog.
    """

    data_source: DataSource
    file_suffix: str
    read: Callable[[Path], TcgjsonCatalog]


# tcgjson's weekly GitHub release (src/data_retrieval/tcgjson/)
TCGJSON_FORMAT = CatalogFormat(DataSource.TCGJSON, ".full.json.gz", read_catalog)
