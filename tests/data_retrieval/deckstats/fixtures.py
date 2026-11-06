"""Shared builders for the deckstats retrieval tests."""

import json
from typing import Any


def page_html(page: dict[str, Any]) -> str:
    """A Deckstats page carrying page as its Inertia page JSON."""
    return (
        "<html><head><title>x</title></head><body>"
        f'<script data-page="app" type="application/json">{json.dumps(page)}</script>'
        "</body></html>"
    )


def deck_page(deck_id: int, name: str = "Burn") -> dict[str, Any]:
    """A "Deck" page in the site's shape (2026-11-06), with owner, price
    and card-text keys that pruning drops."""
    return {
        "component": "Deck",
        "url": f"/decks/{deck_id}-{name}",
        "version": "v1",
        "props": {
            "ads": {"slot": 1},
            "priceSources": {"sources": []},
            "deck": {
                "idsdeck": deck_id,
                "name": name,
                "format_name": "Casual",
                "visibility": "public",
                "is_public": True,
                "added": 1595510733,
                "owner_id": 57455,
                "owner_name": "someone",
                "owner_avatar": "/img/a.png",
                "legacy_url": f"https://deckstats.net/decks/57455/{deck_id}/",
                "cheapest_prices": {"usd": 1},
            },
            "sections": [{"section_position": 1, "name": "Main", "comment": ""}],
            "entries": [
                {
                    "zone": "main",
                    "section_position": 1,
                    "card_position": 1,
                    "amount": 4,
                    "name": "Lightning Bolt",
                    "is_commander": False,
                    "is_companion": False,
                    "is_foil": False,
                    "is_proxy": False,
                    "is_game_changer": False,
                    "comment": "",
                    "idcard": 10,
                    "idprintings": 20,
                    "card": {
                        "idcards": 30,
                        "name": "Lightning Bolt",
                        "layout": "normal",
                        "is_token": False,
                        "text": "dropped",
                    },
                    "printing": {
                        "idprintings": 20,
                        "set_abbr": "M10",
                        "collector_number": "146",
                        "scryfall_id": "e3285e6b-3e79-4d7c-bf96-d920f973b80c",
                        "artist": "dropped",
                    },
                }
            ],
            "tags": ["burn"],
        },
    }


def private_page(deck_id: int, deck_name: str = "Secret") -> dict[str, Any]:
    """A "DeckPrivate" page: the deck's name only."""
    return {
        "component": "DeckPrivate",
        "url": f"/decks/{deck_id}",
        "version": "v1",
        "props": {"deckName": deck_name, "meta": {"title": "Private deck"}},
    }
