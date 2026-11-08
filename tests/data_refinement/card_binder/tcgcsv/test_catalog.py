from pathlib import Path

import pytest

from src.data_refinement.card_binder.tcgcsv.catalog import (
    TCGCSV_FORMAT,
    read_tcgcsv_catalog,
)
from src.data_retrieval.tcgcsv.downloader import CATALOG_FILE_SUFFIX
from src.schema.data_source import DataSource
from tests.data_refinement.card_binder.tcgcsv.fixtures import (
    product,
    unit,
    write_catalog,
)


def test_groups_become_sets_and_extended_data_the_attributes(tmp_path: Path) -> None:
    path = write_catalog(tmp_path / "gundam.json.gz", [unit(10, "Gundam", "GD01-001")])
    catalog = read_tcgcsv_catalog(path)

    assert catalog.sets[1].name == "Newtype Rising"
    assert catalog.sets[1].release_date == "2025-07-11T00:00:00"
    assert catalog.sets[2].release_date is None
    (gundam,) = catalog.products
    assert (gundam.product_id, gundam.name, gundam.set_id) == (10, "Gundam", 1)
    assert gundam.rarity == "Common"
    assert gundam.card_types == ("Unit",)
    assert gundam.rules_text == "[Blocker]<br>\r\n(Rest this Unit.)"
    assert gundam.attributes["number"] == "GD01-001"
    assert "Number" not in gundam.attributes
    assert gundam.attributes["Attack Points"] == "3"


def test_a_sealed_product_has_no_type_rarity_or_text(tmp_path: Path) -> None:
    path = write_catalog(tmp_path / "g.json.gz", [product(5, "Booster Box")])
    (box,) = read_tcgcsv_catalog(path).products
    assert (box.card_types, box.rarity, box.rules_text) == ((), "", None)
    assert dict(box.attributes) == {}


def test_malformed_extended_data_entries_are_skipped(tmp_path: Path) -> None:
    raw = product(5, "Odd")
    raw["extendedData"] = [{"value": "x"}, "junk", {"name": "Cost", "value": "2"}]
    (odd,) = read_tcgcsv_catalog(write_catalog(tmp_path / "g.json.gz", [raw])).products
    assert dict(odd.attributes) == {"Cost": "2"}


def test_a_product_in_an_unknown_group_raises(tmp_path: Path) -> None:
    path = write_catalog(tmp_path / "g.json.gz", [product(5, "Lost", group_id=9)])
    with pytest.raises(KeyError):
        read_tcgcsv_catalog(path)


def test_a_catalog_without_groups_raises(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    path.write_text('{"products": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="groups"):
        read_tcgcsv_catalog(path)


def test_the_format_names_the_landed_files_and_tcgcsv() -> None:
    assert TCGCSV_FORMAT.file_suffix == CATALOG_FILE_SUFFIX
    assert TCGCSV_FORMAT.data_source is DataSource.TCGCSV
    assert TCGCSV_FORMAT.read is read_tcgcsv_catalog
