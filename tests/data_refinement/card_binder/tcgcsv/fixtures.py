"""Small tcgcsv catalogs for tests, shaped like the real <slug>.json.gz
(snapshot 2026-11-06)."""

import gzip
import json
from pathlib import Path
from typing import Any

NEWTYPE_RISING = {
    "groupId": 1,
    "name": "Newtype Rising",
    "abbreviation": "GD01",
    "publishedOn": "2025-07-11T00:00:00",
}
PROMOS = {"groupId": 2, "name": "Promos", "abbreviation": "PR"}
GROUPS = [NEWTYPE_RISING, PROMOS]


def product(product_id: int, name: str, group_id: int = 1, **data: str) -> dict:
    """One products[] entry; data become extendedData entries, keyword
    underscores read as spaces (Attack_Points="3")."""
    return {
        "productId": product_id,
        "name": name,
        "cleanName": name,
        "imageUrl": f"https://cdn.test/{product_id}.jpg",
        "groupId": group_id,
        "extendedData": [
            {"name": key.replace("_", " "), "displayName": key, "value": value}
            for key, value in data.items()
        ],
    }


def unit(product_id: int, name: str, number: str, **data: str) -> dict:
    """A Gundam Unit printing with the usual fields."""
    fields = {
        "Rarity": "Common",
        "Number": number,
        "CardType": "Unit",
        "Description": "[Blocker]<br>\r\n(Rest this Unit.)",
        "Level": "3",
        "Cost": "2",
        "Color": "Blue",
        "Attack_Points": "3",
        "Hit_Points": "4",
        **data,
    }
    return product(product_id, name, **fields)


def write_catalog(
    path: Path, products: list[dict[str, Any]], groups: list[dict] = GROUPS
) -> Path:
    """Write a gzipped catalog to path; return path."""
    payload = {"category_id": 86, "groups": groups, "products": products}
    path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))
    return path
