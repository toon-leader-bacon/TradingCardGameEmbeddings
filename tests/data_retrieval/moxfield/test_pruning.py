from typing import Any

import pytest

from src.data_retrieval.moxfield.pruning import prune_deck
from tests.data_retrieval.moxfield.fixtures import deck_payload

_ID = "KtNrenwexES3cLg2NDMEnw"


class TestPruneDeck:
    def test_keeps_api_names_and_drops_owners_bulk_and_empty_boards(self) -> None:
        assert prune_deck(deck_payload(_ID)) == {
            "publicId": _ID,
            "id": "LzdX5",
            "name": "Atraxa Control",
            "format": "commander",
            "visibility": "public",
            "bracket": 4,
            "boards": {
                "mainboard": {
                    "count": 2,
                    "cards": {
                        "kaN49": {
                            "quantity": 2,
                            "boardType": "mainboard",
                            "finish": "nonFoil",
                            "isProxy": False,
                            "card": {
                                "id": "kpard",
                                "uniqueCardId": "kaN49",
                                "scryfall_id": "105b2118-b22c-4ef5-bac7-836db4b8b9ee",
                                "name": "Island",
                                "set": "ust",
                                "cn": "213",
                                "layout": "normal",
                                "lang": "en",
                            },
                        }
                    },
                }
            },
        }

    def test_docstring_example(self) -> None:
        payload = {
            "publicId": "x",
            "boards": {"mainboard": {"count": 0, "cards": {}}},
            "createdByUser": {"userName": "y"},
        }
        assert prune_deck(payload) == {"publicId": "x", "boards": {}}

    @pytest.mark.parametrize(
        "payload, error",
        [
            ([], TypeError),
            ({"boards": {}}, KeyError),
            ({"publicId": "x"}, KeyError),
            ({"publicId": "x", "boards": []}, TypeError),
            ({"publicId": "x", "boards": {"m": {"count": 1}}}, KeyError),
            ({"publicId": "x", "boards": {"m": {"cards": []}}}, TypeError),
            ({"publicId": "x", "boards": {"m": {"cards": {"k": {}}}}}, KeyError),
            (
                {"publicId": "x", "boards": {"m": {"cards": {"k": {"card": 1}}}}},
                TypeError,
            ),
        ],
    )
    def test_unreadable_decks_raise(self, payload: Any, error: type) -> None:
        with pytest.raises(error):
            prune_deck(payload)
