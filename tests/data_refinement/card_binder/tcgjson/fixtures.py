"""Small tcgjson catalogs for tests, shaped like the real
<slug>.full.json.gz (release weekly-20260831)."""

import gzip
import json
from pathlib import Path
from typing import Any

FIRST_CHAPTER = {
    "setId": 1,
    "name": "The First Chapter",
    "releaseDate": "2023-08-18T00:00:00",
}
PROMOS = {"setId": 2, "name": "D23 Promos", "releaseDate": "2022-09-09T00:00:00"}
QUEST = {
    "setId": 3,
    "name": "Illumineer's Quest: Deep Trouble",
    "releaseDate": "2024-11-15T00:00:00",
}
SETS = [FIRST_CHAPTER, PROMOS, QUEST]


def product(
    product_id: int,
    name: str,
    set_id: int = 1,
    rarity: str = "Common",
    card_types: list[str] | None = None,
    rules_text: str | None = "<strong>Evasive</strong>",
    **attributes: Any,
) -> dict[str, Any]:
    """One products[] entry; attributes are customAttributes (costInk="8")."""
    metadata: dict[str, Any] = {
        "cardTypes": ["Character"] if card_types is None else card_types,
        "customAttributes": {"inkType": "Ruby", "costInk": "3", **attributes},
    }
    if rules_text is not None:
        metadata["rulesText"] = rules_text
    return {
        "productId": product_id,
        "name": name,
        "setId": set_id,
        "rarity": rarity,
        "imageUrls": [f"https://cdn.test/{product_id}.jpg"],
        "metadata": metadata,
    }


def write_catalog(
    path: Path, products: list[dict[str, Any]], sets: list[dict[str, Any]] = SETS
) -> Path:
    """Write a gzipped catalog to path; return path."""
    payload = {"meta": {"slug": "lorcana"}, "sets": sets, "products": products}
    path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))
    return path
