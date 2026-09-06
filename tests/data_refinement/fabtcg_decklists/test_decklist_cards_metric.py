import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pyarrow.parquet as pq
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.fabtcg_decklists.decklist_cards_metric import (
    DecklistCardsMetric,
    scan_decklists_dir,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_REAL_FRAGMENT_PATH = Path("data/tmp/list_view.html")


def _binder_with_cards(names: list[str]) -> CardBinder:
    binder = CardBinder()
    for name in names:
        binder.create(
            GenericCard(
                nocab_uuid=uuid4(),
                source_game=GameId.FLESH_AND_BLOOD,
                name=name,
                raw_content={},
                provenance=Provenance(
                    data_source=DataSource.CARDVAULT_FABTCG,
                    source_id=name,
                    fetched_at=datetime.now(timezone.utc),
                ),
            )
        )
    return binder


def _uuid_for(binder: CardBinder, name: str) -> UUID:
    card = binder.get_by_name_single(GameId.FLESH_AND_BLOOD, name)
    assert card is not None
    return card.nocab_uuid


def _fragment(*groups: tuple[str, list[str]]) -> str:
    """Build a synthetic decklist fragment: groups is
    (header_text, [card_name_div_html, ...]) pairs."""
    group_html = "".join(
        f'<div class="list-view-container">'
        f"<h3>{header}</h3>"
        f'<ul class="cards-container">{"".join(items)}</ul>'
        f"</div>"
        for header, items in groups
    )
    return f'<section class="decklist-list-view block hidden">{group_html}</section>'


def _card_item(quantity_and_name: str) -> str:
    return (
        '<li class="card-item group">'
        f'<div class="card-name"><span>{quantity_and_name}</span></div>'
        "</li>"
    )


class TestParseQuantityAndName:
    def test_single_digit_quantity(self, tmp_path: Path) -> None:
        binder = _binder_with_cards([])
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")
        assert metric._parse_quantity_and_name("1x Dorinthea Ironsong") == (
            1,
            "Dorinthea Ironsong",
        )
        metric.finalize()

    def test_multi_digit_quantity(self, tmp_path: Path) -> None:
        binder = _binder_with_cards([])
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")
        assert metric._parse_quantity_and_name("12x Some Card") == (12, "Some Card")
        metric.finalize()

    def test_raises_on_non_matching_text(self, tmp_path: Path) -> None:
        binder = _binder_with_cards([])
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")
        with pytest.raises(ValueError):
            metric._parse_quantity_and_name("no quantity prefix here")
        metric.finalize()


class TestExtractCardUuids:
    def test_walks_hero_and_pitch_groups_and_preserves_multi_copy(
        self, tmp_path: Path
    ) -> None:
        binder = _binder_with_cards(["Dorinthea Ironsong", "Enlightened Strike"])
        fragment = _fragment(
            ("Hero / Weapon / Equipment", [_card_item("1x Dorinthea Ironsong")]),
            ("Pitch 1", [_card_item("3x Enlightened Strike")]),
        )
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")

        card_uuids = metric._extract_card_uuids(fragment)

        hero_uuid = _uuid_for(binder, "Dorinthea Ironsong")
        pitch_uuid = _uuid_for(binder, "Enlightened Strike")
        assert card_uuids == [hero_uuid, pitch_uuid, pitch_uuid, pitch_uuid]
        metric.finalize()

    def test_unresolved_card_is_dropped_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _binder_with_cards(["Dorinthea Ironsong"])
        fragment = _fragment(
            (
                "Hero / Weapon / Equipment",
                [
                    _card_item("1x Dorinthea Ironsong"),
                    _card_item("1x Not A Real Card"),
                ],
            )
        )
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")

        with caplog.at_level(logging.WARNING):
            card_uuids = metric._extract_card_uuids(fragment)

        assert card_uuids == [_uuid_for(binder, "Dorinthea Ironsong")]
        assert any(
            record.levelno == logging.WARNING
            and "Not A Real Card" in record.getMessage()
            for record in caplog.records
        )
        metric.finalize()

    def test_raises_when_card_item_missing_name_div(self, tmp_path: Path) -> None:
        binder = _binder_with_cards([])
        fragment = (
            '<section class="decklist-list-view block hidden">'
            '<div class="list-view-container"><h3>Hero</h3>'
            '<ul class="cards-container">'
            '<li class="card-item group"><div class="card-image"></div></li>'
            "</ul></div></section>"
        )
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")

        with pytest.raises(ValueError):
            metric._extract_card_uuids(fragment)
        metric.finalize()

    def test_against_real_fixture(self, tmp_path: Path) -> None:
        if not _REAL_FRAGMENT_PATH.exists():
            pytest.skip(f"{_REAL_FRAGMENT_PATH} fixture not present")

        fragment_html = _REAL_FRAGMENT_PATH.read_text(encoding="utf-8")
        binder = _binder_with_cards(["Dorinthea Ironsong", "Enlightened Strike (red)"])
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")

        card_uuids = metric._extract_card_uuids(fragment_html)

        hero_uuid = _uuid_for(binder, "Dorinthea Ironsong")
        pitch_uuid = _uuid_for(binder, "Enlightened Strike (red)")
        assert card_uuids.count(hero_uuid) == 1
        assert card_uuids.count(pitch_uuid) == 3
        metric.finalize()


class TestInit:
    def test_creates_parent_dir_and_opens_writer(self, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "decklists.parquet"
        binder = _binder_with_cards([])

        metric = DecklistCardsMetric(binder, output_path)
        metric.finalize()

        assert output_path.exists()


class TestAccumulate:
    def test_writes_a_row_readable_after_finalize(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Dorinthea Ironsong"])
        fragment = _fragment(
            ("Hero / Weapon / Equipment", [_card_item("1x Dorinthea Ironsong")])
        )
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")

        metric.accumulate("some-deck-slug", fragment)
        metric.finalize()

        rows = pq.read_table(tmp_path / "out.parquet").to_pylist()
        assert rows == [
            {
                "deck_slug": "some-deck-slug",
                "card_nocab_uuids": [str(_uuid_for(binder, "Dorinthea Ironsong"))],
            }
        ]

    def test_raises_when_zero_cards_resolve(self, tmp_path: Path) -> None:
        binder = _binder_with_cards([])
        fragment = _fragment(
            ("Hero / Weapon / Equipment", [_card_item("1x Not A Real Card")])
        )
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")

        with pytest.raises(ValueError):
            metric.accumulate("empty-deck", fragment)
        metric.finalize()


class TestFinalize:
    def test_is_idempotent(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Dorinthea Ironsong"])
        fragment = _fragment(
            ("Hero / Weapon / Equipment", [_card_item("1x Dorinthea Ironsong")])
        )
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")
        metric.accumulate("some-deck-slug", fragment)

        first_path = metric.finalize()
        second_path = metric.finalize()

        assert first_path == second_path == tmp_path / "out.parquet"
        assert pq.read_table(first_path).num_rows == 1


class TestScanDecklistsDir:
    def test_drives_every_file_into_one_parquet_output(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Dorinthea Ironsong", "Enlightened Strike"])
        decklists_dir = tmp_path / "decklists"
        decklists_dir.mkdir()
        (decklists_dir / "deck-one.html").write_text(
            _fragment(
                ("Hero / Weapon / Equipment", [_card_item("1x Dorinthea Ironsong")])
            ),
            encoding="utf-8",
        )
        (decklists_dir / "deck-two.html").write_text(
            _fragment(("Pitch 1", [_card_item("2x Enlightened Strike")])),
            encoding="utf-8",
        )
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")

        output_path = scan_decklists_dir(decklists_dir, metric)

        rows = pq.read_table(output_path).to_pylist()
        assert [row["deck_slug"] for row in rows] == ["deck-one", "deck-two"]

    def test_still_finalizes_when_a_file_raises(self, tmp_path: Path) -> None:
        binder = _binder_with_cards(["Dorinthea Ironsong"])
        decklists_dir = tmp_path / "decklists"
        decklists_dir.mkdir()
        (decklists_dir / "good-deck.html").write_text(
            _fragment(
                ("Hero / Weapon / Equipment", [_card_item("1x Dorinthea Ironsong")])
            ),
            encoding="utf-8",
        )
        (decklists_dir / "zzz-bad-deck.html").write_text(
            _fragment(
                ("Hero / Weapon / Equipment", [_card_item("1x Not A Real Card")])
            ),
            encoding="utf-8",
        )
        metric = DecklistCardsMetric(binder, tmp_path / "out.parquet")

        with pytest.raises(ValueError):
            scan_decklists_dir(decklists_dir, metric)

        rows = pq.read_table(tmp_path / "out.parquet").to_pylist()
        assert [row["deck_slug"] for row in rows] == ["good-deck"]
