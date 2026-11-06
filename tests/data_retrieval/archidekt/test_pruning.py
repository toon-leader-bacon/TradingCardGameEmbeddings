from typing import Any

import pytest

from src.data_retrieval.archidekt.pruning import prune_deck
from tests.data_retrieval.archidekt.fixtures import deck_payload


class TestPruneDeck:
    def test_keeps_api_names_and_drops_bulk_owner_and_binder_fields(self) -> None:
        row = prune_deck(deck_payload(4))
        assert row == {
            "id": 4,
            "name": "Deck 4",
            "deckFormat": 3,
            "unlisted": True,
            "categories": [{"name": "Commander", "includedInDeck": True}],
            "cards": [
                {
                    "quantity": 2,
                    "modifier": "Normal",
                    "categories": ["Commander"],
                    "card": {
                        "collectorNumber": "111",
                        "edition": {"editioncode": "akh"},
                        "oracleCard": {"uid": "o-1", "name": "Card"},
                    },
                }
            ],
        }

    def test_docstring_example(self) -> None:
        payload = {"id": 1, "cards": [], "owner": {"id": 9}, "game": None}
        assert prune_deck(payload) == {"id": 1, "cards": []}

    def test_null_values_are_left_out_at_every_level(self) -> None:
        payload = deck_payload(3)
        payload["description"] = None
        payload["cards"][0].update({"notes": None, "deletedAt": None, "label": None})
        payload["cards"][0]["card"]["oracleCard"]["layout"] = None
        row = prune_deck(payload)
        assert "description" not in row
        assert set(row["cards"][0]) == {"quantity", "modifier", "categories", "card"}
        assert "layout" not in row["cards"][0]["card"]["oracleCard"]

    def test_falsy_values_that_are_not_null_are_kept(self) -> None:
        payload = deck_payload(3)
        payload["cards"][0].update({"companion": False, "notes": ""})
        row = prune_deck(payload)["cards"][0]
        assert row["companion"] is False and row["notes"] == ""

    def test_a_row_pruned_the_old_way_prunes_to_the_new_shape(self) -> None:
        old_row = prune_deck(deck_payload(4))
        old_row["cards"][0].update({"id": 11, "createdAt": "2022-06-08"})
        old_row["cards"][0]["card"].update({"uid": "p-1", "releasedAt": "2017-04-28"})
        old_row["cards"][0]["card"]["edition"]["editionname"] = "Amonkhet"
        assert prune_deck(old_row) == prune_deck(deck_payload(4))

    @pytest.mark.parametrize(
        "payload, error",
        [
            ({"cards": []}, KeyError),
            ({"id": 1}, KeyError),
            ({"id": 1, "cards": {}}, TypeError),
            ({"id": 1, "cards": [{"quantity": 1}]}, KeyError),
            ({"id": 1, "cards": [{"card": None}]}, TypeError),
            ([], TypeError),
        ],
    )
    def test_unreadable_decks_raise(self, payload: Any, error: type) -> None:
        with pytest.raises(error):
            prune_deck(payload)
