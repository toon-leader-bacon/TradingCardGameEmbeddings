from pathlib import Path
from typing import Any

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.tcgjson.game_spec import ALL_SPECS, LORCANA_SPEC
from src.data_refinement.card_binder.tcgjson.ingestion_stage import (
    LorcanaTcgjsonCardIngestionStage,
    OnePieceTcgjsonCardIngestionStage,
    TcgjsonCardIngestionStage,
)
from src.data_retrieval.tcgjson.downloader import CATALOG_SLUGS
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.card_binder.tcgjson.fixtures import product, write_catalog

_LORCANA = GameId.LORCANA


def _ingest(
    tmp_path: Path,
    products: list[dict[str, Any]],
    binder: CardBinder,
    name: str = "lorcana.full.json.gz",
) -> list:
    path = write_catalog(tmp_path / name, products)
    return LorcanaTcgjsonCardIngestionStage().ingest(path, binder)


def _alias(binder: CardBinder, product_id: int) -> Any:
    return binder.get_by_alias(_LORCANA, DataSource.TCGJSON, str(product_id))


def test_source_game_comes_from_the_spec() -> None:
    assert LorcanaTcgjsonCardIngestionStage.SOURCE_GAME is GameId.LORCANA
    assert LorcanaTcgjsonCardIngestionStage.SPEC is LORCANA_SPEC


def test_every_game_has_one_stage_and_its_own_game() -> None:
    stages = TcgjsonCardIngestionStage.__subclasses__()
    games = [stage.SOURCE_GAME for stage in stages]
    assert len(stages) == 7
    assert len(set(games)) == len(games)
    assert all(stage.SOURCE_GAME is stage.SPEC.game for stage in stages)
    assert {stage.SPEC for stage in stages} == set(ALL_SPECS)


def test_the_downloader_fetches_exactly_the_specs_catalogs() -> None:
    assert set(CATALOG_SLUGS) == {spec.slug for spec in ALL_SPECS}


def test_a_subclass_without_a_spec_is_refused() -> None:
    with pytest.raises(TypeError):

        class _NoSpec(TcgjsonCardIngestionStage):  # noqa: F841
            pass


def test_one_card_per_printing_group_with_every_product_id_as_an_alias(
    tmp_path: Path,
) -> None:
    binder = CardBinder.load([])
    changed = _ingest(
        tmp_path,
        [
            product(10, "Elsa - Snow Queen"),
            product(30, "Elsa - Snow Queen (Epic)", rarity="Epic"),
            product(20, "Ariel - On Human Legs"),
        ],
        binder,
    )

    assert len(changed) == 2
    elsa = _alias(binder, 30)
    assert elsa is not None and elsa.name == "Elsa - Snow Queen"
    assert _alias(binder, 10) == elsa
    assert elsa.provenance.data_source is DataSource.TCGJSON
    assert elsa.provenance.source_id == "10"  # Common beats Epic
    assert elsa.raw_content["rarity"] == "Common"


def test_a_release_directory_reads_the_specs_catalog(tmp_path: Path) -> None:
    binder = CardBinder.load([])
    write_catalog(tmp_path / "lorcana.full.json.gz", [product(1, "Elsa")])
    LorcanaTcgjsonCardIngestionStage().ingest(tmp_path, binder)
    assert _alias(binder, 1) is not None


def test_a_reingest_of_the_same_catalog_changes_nothing(tmp_path: Path) -> None:
    binder = CardBinder.load([])
    products = [product(1, "Elsa"), product(2, "Ariel")]
    _ingest(tmp_path, products, binder)
    uuids = set(binder.all_uuids(_LORCANA))

    assert _ingest(tmp_path, products, binder) == []
    assert set(binder.all_uuids(_LORCANA)) == uuids


def test_a_changed_card_keeps_its_uuid(tmp_path: Path) -> None:
    binder = CardBinder.load([])
    _ingest(tmp_path, [product(1, "Elsa", costInk="3")], binder)
    before = _alias(binder, 1)

    changed = _ingest(
        tmp_path,
        [product(1, "Elsa", costInk="4"), product(5, "Elsa (Epic)", costInk="4")],
        binder,
    )

    after = _alias(binder, 5)
    assert changed == [before.nocab_uuid]
    assert after.nocab_uuid == before.nocab_uuid and after.raw_content["cost"] == 4


def test_cards_the_catalog_dropped_are_deleted_but_not_the_sentinel(
    tmp_path: Path,
) -> None:
    binder = CardBinder.load([])
    sentinel = binder.ensure_unknown_card(_LORCANA)
    _ingest(tmp_path, [product(1, "Elsa"), product(2, "Ariel")], binder)

    _ingest(tmp_path, [product(1, "Elsa")], binder)

    names = sorted(card.name for card in binder.all_cards(_LORCANA))
    assert names == sorted(["Elsa", sentinel.name])


def test_a_catalog_without_cards_raises_and_keeps_the_binder(tmp_path: Path) -> None:
    binder = CardBinder.load([])
    _ingest(tmp_path, [product(1, "Elsa")], binder)
    with pytest.raises(ValueError):
        _ingest(tmp_path, [product(2, "Insert", card_types=[])], binder)
    assert _alias(binder, 1) is not None


def test_a_missing_catalog_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LorcanaTcgjsonCardIngestionStage().ingest(tmp_path, CardBinder.load([]))


def test_a_card_number_game_registers_its_numbers_as_aliases(tmp_path: Path) -> None:
    binder = CardBinder.load([])
    path = write_catalog(
        tmp_path / "one-piece.full.json.gz",
        [
            product(1, "Monkey.D.Luffy", number="OP01-024"),
            product(2, "Monkey.D.Luffy (Parallel)", number="OP01-024"),
            product(3, "Monkey.D.Luffy", number="ST01-012"),
        ],
    )
    OnePieceTcgjsonCardIngestionStage().ingest(path, binder)

    first = binder.get_by_alias(GameId.ONE_PIECE, DataSource.PRINTED_NUMBER, "OP01-024")
    second = binder.get_by_alias(
        GameId.ONE_PIECE, DataSource.PRINTED_NUMBER, "ST01-012"
    )
    assert first is not None and second is not None
    assert first.nocab_uuid != second.nocab_uuid
    assert first == binder.get_by_alias(GameId.ONE_PIECE, DataSource.TCGJSON, "2")


def test_a_name_game_registers_no_numbers(tmp_path: Path) -> None:
    binder = CardBinder.load([])
    _ingest(tmp_path, [product(1, "Elsa - Snow Queen", number="1/204")], binder)
    assert binder.get_by_alias(_LORCANA, DataSource.PRINTED_NUMBER, "1/204") is None
