"""Shared builders for the moxfield retrieval tests."""

from typing import Any


def deck_payload(public_id: str) -> dict[str, Any]:
    """A v3/decks/all payload in the API's shape (2026-11-05), with bulk
    and owner keys that pruning drops."""
    return {
        "id": "LzdX5",
        "publicId": public_id,
        "name": "Atraxa Control",
        "format": "commander",
        "visibility": "public",
        "bracket": 4,
        "createdByUser": {"userName": "someone", "displayName": "Someone"},
        "authors": [{"userName": "someone"}],
        "ownerUserId": "kpxQB",
        "tokens": [{"name": "Zombie"}],
        "boards": {
            "mainboard": {
                "count": 2,
                "cards": {
                    "kaN49": {
                        "quantity": 2,
                        "boardType": "mainboard",
                        "finish": "nonFoil",
                        "isProxy": False,
                        "isFoil": False,
                        "card": {
                            "id": "kpard",
                            "uniqueCardId": "kaN49",
                            "scryfall_id": "105b2118-b22c-4ef5-bac7-836db4b8b9ee",
                            "name": "Island",
                            "set": "ust",
                            "cn": "213",
                            "layout": "normal",
                            "lang": "en",
                            "oracle_text": "dropped",
                            "prices": {"usd": 1.0},
                            "cardKingdomUrl": "dropped",
                        },
                    }
                },
            },
            "sideboard": {"count": 0, "cards": {}},
        },
    }
