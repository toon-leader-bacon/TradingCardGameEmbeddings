import json
from pathlib import Path

import pytest

from src.data_refinement.card_binder.tcgjson.catalog import read_catalog
from tests.data_refinement.card_binder.tcgjson.fixtures import product, write_catalog


def test_reads_sets_and_products(tmp_path: Path) -> None:
    path = write_catalog(
        tmp_path / "lorcana.full.json.gz",
        [product(7, "Elsa - Snow Queen", rarity="Uncommon", costInk="3")],
    )
    catalog = read_catalog(path)

    assert catalog.sets[1].name == "The First Chapter"
    assert catalog.sets[1].release_date == "2023-08-18T00:00:00"
    (elsa,) = catalog.products
    assert (elsa.product_id, elsa.name, elsa.set_id, elsa.rarity) == (
        7,
        "Elsa - Snow Queen",
        1,
        "Uncommon",
    )
    assert elsa.card_types == ("Character",)
    assert elsa.rules_text == "<strong>Evasive</strong>"
    assert elsa.attributes["costInk"] == "3"


def test_reads_plain_json(tmp_path: Path) -> None:
    path = tmp_path / "lorcana.full.json"
    path.write_text(json.dumps({"sets": [], "products": []}))
    assert read_catalog(path).products == ()


def test_a_product_without_metadata_reads_as_empty(tmp_path: Path) -> None:
    raw = {"productId": 1, "name": "Puzzle Insert (Set of 4)", "setId": 1}
    path = write_catalog(tmp_path / "c.json.gz", [raw])
    (insert,) = read_catalog(path).products
    assert insert.card_types == () and insert.rules_text is None
    assert dict(insert.attributes) == {} and insert.rarity == ""


def test_a_set_without_a_release_date_reads_as_none(tmp_path: Path) -> None:
    path = write_catalog(tmp_path / "c.json.gz", [], [{"setId": 9, "name": "S"}])
    assert read_catalog(path).sets[9].release_date is None


def test_attributes_are_read_only(tmp_path: Path) -> None:
    path = write_catalog(tmp_path / "c.json.gz", [product(1, "A")])
    with pytest.raises(TypeError):
        read_catalog(path).products[0].attributes["costInk"] = "9"  # type: ignore[index]


@pytest.mark.parametrize(
    "raw",
    [
        {"name": "No id", "setId": 1},
        {"productId": "1", "name": "String id", "setId": 1},
        {"productId": 1, "setId": 1},
        {"productId": 1, "name": "", "setId": 1},
        {"productId": 1, "name": "Unknown set", "setId": 99},
    ],
)
def test_a_malformed_product_raises(tmp_path: Path, raw: dict[str, object]) -> None:
    path = write_catalog(tmp_path / "c.json.gz", [raw])
    with pytest.raises((KeyError, TypeError)):
        read_catalog(path)


def test_a_catalog_without_products_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"sets": []}))
    with pytest.raises(ValueError):
        read_catalog(path)


def test_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_catalog(tmp_path / "absent.json.gz")


def test_rules_text_falls_back_to_the_description_attribute(tmp_path: Path) -> None:
    path = write_catalog(
        tmp_path / "c.json.gz",
        [
            product(1, "Agumon", rules_text=None, description="[On Play] Draw 1."),
            product(2, "Gabumon", rules_text="Kept.", description="Ignored."),
        ],
    )
    agumon, gabumon = read_catalog(path).products
    assert agumon.rules_text == "[On Play] Draw 1."
    assert gabumon.rules_text == "Kept."
