from typing import Any

import pytest

from src.data_retrieval.deckstats.page import parse_inertia_page
from src.data_retrieval.deckstats.pruning import (
    PRIVATE_SUBDIRECTORY,
    build_private_deck_row,
    classify_page,
    prune_deck,
)
from src.data_retrieval.resumable_crawl import ItemFound, ItemRejected
from tests.data_retrieval.deckstats.fixtures import deck_page, page_html, private_page


class TestParseInertiaPage:
    def test_reads_the_page_json(self) -> None:
        assert parse_inertia_page(page_html(private_page(5)))["component"] == (
            "DeckPrivate"
        )

    @pytest.mark.parametrize(
        "html",
        [
            "<html>Just a moment...</html>",
            '<script data-page="app" type="application/json">{not json</script>',
            '<script data-page="app" type="application/json">[1, 2]</script>',
        ],
    )
    def test_a_page_without_readable_page_json_raises(self, html: str) -> None:
        with pytest.raises(ValueError):
            parse_inertia_page(html)


class TestClassifyPage:
    def test_a_deck_page_is_found_and_pruned(self) -> None:
        outcome = classify_page(deck_page(7))
        assert isinstance(outcome, ItemFound) and outcome.subdirectory is None
        assert outcome.row["deck"]["idsdeck"] == 7

    def test_a_private_page_lands_in_the_private_subdirectory(self) -> None:
        assert classify_page(private_page(5, "Einkaufsliste")) == ItemFound(
            {"idsdeck": 5, "deckName": "Einkaufsliste", "visibility": "private"},
            PRIVATE_SUBDIRECTORY,
        )

    def test_another_component_is_rejected(self) -> None:
        assert classify_page({"component": "Error", "props": {}}) == ItemRejected(
            "page Error"
        )

    @pytest.mark.parametrize(
        "page",
        [
            {},
            {"component": "Deck"},
            {"component": "DeckPrivate", "props": {"deckName": "x"}},
            {"component": "DeckPrivate", "url": "/decks/x", "props": {"deckName": "x"}},
            {"component": "DeckPrivate", "url": 5, "props": {"deckName": "x"}},
        ],
    )
    def test_a_malformed_page_raises_a_reject_type(self, page: dict[str, Any]) -> None:
        # read_outcome rejects KeyError/TypeError; a ValueError would be transient
        with pytest.raises((KeyError, TypeError)):
            classify_page(page)


class TestPruneDeck:
    def test_keeps_the_site_names_and_drops_owner_prices_and_card_text(self) -> None:
        assert prune_deck(deck_page(7)["props"]) == {
            "deck": {
                "idsdeck": 7,
                "name": "Burn",
                "format_name": "Casual",
                "visibility": "public",
                "is_public": True,
                "added": 1595510733,
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
                    "comment": "",
                    "idcard": 10,
                    "idprintings": 20,
                    "card": {"idcards": 30, "layout": "normal", "is_token": False},
                    "printing": {
                        "set_abbr": "M10",
                        "collector_number": "146",
                        "scryfall_id": "e3285e6b-3e79-4d7c-bf96-d920f973b80c",
                    },
                }
            ],
            "tags": ["burn"],
        }

    def test_docstring_example(self) -> None:
        props = {"deck": {"idsdeck": 1, "owner_name": "x"}, "entries": [], "ads": {}}
        assert prune_deck(props) == {"deck": {"idsdeck": 1}, "entries": []}

    @pytest.mark.parametrize(
        "props, error",
        [
            ([], TypeError),
            ({"entries": []}, KeyError),
            ({"deck": {}, "entries": []}, KeyError),
            ({"deck": {"idsdeck": 1}}, KeyError),
            ({"deck": {"idsdeck": 1}, "entries": {}}, TypeError),
            ({"deck": {"idsdeck": 1}, "entries": [5]}, TypeError),
        ],
    )
    def test_unreadable_props_raise(self, props: Any, error: type) -> None:
        with pytest.raises(error):
            prune_deck(props)


def test_private_row_reads_a_slugged_url() -> None:
    assert (
        build_private_deck_row({"deckName": "x"}, "/decks/9-Old-Name")["idsdeck"] == 9
    )
