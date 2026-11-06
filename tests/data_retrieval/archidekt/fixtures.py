"""Shared builders for the archidekt retrieval tests."""

from typing import Any


def deck_payload(deck_id: int, oracle_uid: str = "o-1") -> dict[str, Any]:
    return {
        "id": deck_id,
        "name": f"Deck {deck_id}",
        "deckFormat": 3,
        "unlisted": deck_id % 2 == 0,
        "owner": {"id": 907, "username": "someone"},
        "featured": "https://img",
        "categories": [{"name": "Commander", "includedInDeck": True}],
        "cards": [
            {
                "id": 11,
                "quantity": 2,
                "modifier": "Normal",
                "categories": ["Commander"],
                "card": {
                    "uid": "p-1",
                    "collectorNumber": "111",
                    "artist": "dropped",
                    "prices": {"ck": 1.0},
                    "edition": {
                        "editioncode": "akh",
                        "editionname": "Amonkhet",
                        "x": 1,
                    },
                    "oracleCard": {
                        "uid": oracle_uid,
                        "name": "Card",
                        "text": "dropped",
                    },
                },
            }
        ],
    }
