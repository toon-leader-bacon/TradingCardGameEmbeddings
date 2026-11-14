from pathlib import Path
from typing import Any

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.tcgcsv.catalog import TCGCSV_FORMAT
from src.data_refinement.card_binder.tcgcsv.game_spec import ALL_SPECS
from src.data_refinement.card_binder.tcgcsv.ingestion_stage import (
    DragonBallMastersTcgcsvCardIngestionStage,
    GundamTcgcsvCardIngestionStage,
)
from src.data_refinement.card_binder.tcgjson.ingestion_stage import (
    TcgjsonCardIngestionStage,
)
from src.data_retrieval.tcgcsv.downloader import CATEGORY_IDS
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.card_binder.tcgcsv.fixtures import (
    product,
    unit,
    write_catalog,
)


def _tcgcsv_stages() -> list[type[TcgjsonCardIngestionStage]]:
    return [
        stage
        for stage in TcgjsonCardIngestionStage.__subclasses__()
        if stage.SPEC.catalog_format is TCGCSV_FORMAT
    ]


def test_every_game_has_one_stage_and_its_own_game() -> None:
    stages = _tcgcsv_stages()
    games = [stage.SOURCE_GAME for stage in stages]
    assert len(stages) == 4
    assert len(set(games)) == len(games)
    assert {stage.SPEC for stage in stages} == set(ALL_SPECS)


def test_the_downloader_fetches_exactly_the_specs_catalogs() -> None:
    assert set(CATEGORY_IDS) == {spec.slug for spec in ALL_SPECS}


def _ingest_gundam(tmp_path: Path, products: list[dict[str, Any]]) -> CardBinder:
    binder = CardBinder.load([])
    write_catalog(tmp_path / "gundam.json.gz", products)
    GundamTcgcsvCardIngestionStage().ingest(tmp_path, binder)  # a snapshot dir
    return binder


def _gundam(binder: CardBinder, product_id: int) -> Any:
    return binder.get_by_alias(GameId.GUNDAM, DataSource.TCGCSV, str(product_id))


def test_a_card_is_its_number_with_tcgcsv_aliases(tmp_path: Path) -> None:
    binder = _ingest_gundam(
        tmp_path,
        [
            unit(10, "Gundam", "GD01-001"),
            unit(11, "Gundam (LR+)", "GD01-001", Rarity="LR+"),
            unit(12, "Gundam", "GD01-002", Cost="5"),
        ],
    )

    gundam = _gundam(binder, 11)
    assert gundam is not None and gundam == _gundam(binder, 10)
    assert gundam != _gundam(binder, 12)
    assert gundam.provenance.data_source is DataSource.TCGCSV
    assert gundam.raw_content == {
        "name": "Gundam",
        "type": "Unit",
        "color": ["Blue"],
        "level": 3,
        "cost": 2,
        "ap": "3",
        "hp": "4",
        "rarity": "Common",
        "set": "Newtype Rising",
        "text": "[Blocker] (Rest this Unit.)",
    }
    number_alias = binder.get_by_alias(
        GameId.GUNDAM, DataSource.PRINTED_NUMBER, "GD01-001"
    )
    assert number_alias == gundam


def test_resources_are_one_card_and_setup_pieces_and_tokens_are_not_cards(
    tmp_path: Path,
) -> None:
    resource = {"Rarity": "Common", "CardType": "Resource"}
    binder = _ingest_gundam(
        tmp_path,
        [
            unit(10, "Gundam", "GD01-001"),
            product(20, "Resource", Number="R-001", **resource),
            product(21, "Resource (R-041)", Number="R-041", **resource),
            product(
                22, "Resource (RP-004) (Anime Expo)", 2, Number="RP-004", **resource
            ),
            product(30, "EX Base", Number="EXB-001", CardType="EX Base"),
            product(31, "EX Resource", Number="EXR-001", CardType="EX Resource"),
            unit(40, "Gozzo (T-030) Token", "T-030"),
            product(50, "Newtype Rising Booster Box"),
        ],
    )

    resource_card = _gundam(binder, 20)
    assert resource_card is not None and resource_card.name == "Resource"
    assert _gundam(binder, 21) == resource_card == _gundam(binder, 22)
    for product_id in (30, 31, 40, 50):
        assert _gundam(binder, product_id) is None
    assert len(binder.all_uuids(GameId.GUNDAM)) == 2


def _masters_product(product_id: int, name: str, number: str, **data: str) -> dict:
    return product(product_id, name, Number=number, Description="Draw 1.", **data)


def test_masters_keeps_untyped_numbered_cards_minus_markers_and_merits(
    tmp_path: Path,
) -> None:
    binder = CardBinder.load([])
    write_catalog(
        tmp_path / "dragon-ball-masters.json.gz",
        [
            _masters_product(10, "Vegito // SSB Vegito", "BT3-055"),
            _masters_product(11, "Launch", "BT31-055", CardType="Leader"),
            _masters_product(12, "Launch (Pre-Release)", "BT31-055_PR"),
            _masters_product(20, "Energy Marker (M-45)", "M-45"),
            _masters_product(21, "Energy Marker (BT31-EM03)", "BT31-EM03"),
            _masters_product(22, "Championship Merit Card", "1"),
            _masters_product(23, "Ghost Token", "P-999", CardType="Token"),
        ],
    )
    DragonBallMastersTcgcsvCardIngestionStage().ingest(tmp_path, binder)

    def card(product_id: int) -> Any:
        return binder.get_by_alias(
            GameId.DRAGON_BALL_MASTERS, DataSource.TCGCSV, str(product_id)
        )

    vegito = card(10)
    assert vegito is not None and "type" not in vegito.raw_content
    launch = card(11)
    assert launch is not None and card(12) == launch
    assert launch.raw_content["type"] == "Leader"
    for product_id in (20, 21, 22, 23):
        assert card(product_id) is None
