import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.fabtcg_decklists.extraction_stage import (
    FabtcgDecklistsExtractionStage,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _fabtcg_card(name: str) -> GenericCard:
    return GenericCard(
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


def _fabtcg_card_binder(names: list[str]) -> CardBinder:
    binder = CardBinder()
    for name in names:
        binder.create(_fabtcg_card(name))
    return binder


def _card_item(quantity: int, name: str) -> str:
    return f'<li class="card-item"><div class="card-name">{quantity}x {name}</div></li>'


def _fragment(card_names_with_quantities: list[tuple[int, str]]) -> str:
    items = "".join(
        _card_item(quantity, name) for quantity, name in card_names_with_quantities
    )
    return (
        '<section class="decklist-list-view block hidden">'
        f'<div class="list-view-container">{items}</div>'
        "</section>"
    )


def _write_fragment(path: Path, deck_slug: str, fragment_html: str) -> None:
    (path / f"{deck_slug}.html").write_text(fragment_html, encoding="utf-8")


class TestExtract:
    def test_all_resolvable_cards_creates_deck(self, tmp_path: Path) -> None:
        binder = _fabtcg_card_binder(["Dorinthea Ironsong", "Command and Conquer"])
        box = DeckBox()
        _write_fragment(
            tmp_path,
            "deck-1",
            _fragment([(1, "Dorinthea Ironsong"), (2, "Command and Conquer")]),
        )
        stage = FabtcgDecklistsExtractionStage()

        changed_uuids = stage.extract(tmp_path, box, binder)

        assert len(changed_uuids) == 1
        deck = box.get_by_uuid(changed_uuids[0])
        assert deck is not None
        assert deck.source_game == GameId.FLESH_AND_BLOOD
        dorinthea_uuid = binder.get_by_name_single(
            GameId.FLESH_AND_BLOOD, "Dorinthea Ironsong"
        ).nocab_uuid
        cnc_uuid = binder.get_by_name_single(
            GameId.FLESH_AND_BLOOD, "Command and Conquer"
        ).nocab_uuid
        assert deck.card_nocab_uuids.count(dorinthea_uuid) == 1
        assert deck.card_nocab_uuids.count(cnc_uuid) == 2

    def test_unresolvable_card_falls_back_to_unknown_sentinel(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _fabtcg_card_binder(["Dorinthea Ironsong"])
        unknown = binder.ensure_unknown_card(GameId.FLESH_AND_BLOOD)
        box = DeckBox()
        _write_fragment(
            tmp_path,
            "deck-1",
            _fragment([(1, "Dorinthea Ironsong"), (1, "Nonexistent Card")]),
        )
        stage = FabtcgDecklistsExtractionStage()

        with caplog.at_level(logging.ERROR):
            changed_uuids = stage.extract(tmp_path, box, binder)

        deck = box.get_by_uuid(changed_uuids[0])
        assert unknown.nocab_uuid in deck.card_nocab_uuids
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_reextracting_same_raw_path_is_idempotent(self, tmp_path: Path) -> None:
        binder = _fabtcg_card_binder(["Dorinthea Ironsong"])
        box = DeckBox()
        _write_fragment(tmp_path, "deck-1", _fragment([(1, "Dorinthea Ironsong")]))
        stage = FabtcgDecklistsExtractionStage()

        first = stage.extract(tmp_path, box, binder)
        second = stage.extract(tmp_path, box, binder)

        assert len(first) == 1
        assert second == []
        assert len(list(box.all_decks(GameId.FLESH_AND_BLOOD))) == 1

    def test_reextracting_after_resolution_changes_updates_not_duplicates(
        self, tmp_path: Path
    ) -> None:
        binder = _fabtcg_card_binder(["Dorinthea Ironsong"])
        unknown = binder.ensure_unknown_card(GameId.FLESH_AND_BLOOD)
        box = DeckBox()
        _write_fragment(
            tmp_path,
            "deck-1",
            _fragment([(1, "Dorinthea Ironsong"), (1, "Command and Conquer")]),
        )
        stage = FabtcgDecklistsExtractionStage()

        first = stage.extract(tmp_path, box, binder)
        assert len(first) == 1
        deck_uuid = first[0]
        deck_after_first = box.get_by_uuid(deck_uuid)
        assert unknown.nocab_uuid in deck_after_first.card_nocab_uuids

        # The CardBinder gains the previously-unresolvable card.
        cnc_card = _fabtcg_card("Command and Conquer")
        binder.create(cnc_card)

        second = stage.extract(tmp_path, box, binder)

        assert second == [deck_uuid]
        assert len(list(box.all_decks(GameId.FLESH_AND_BLOOD))) == 1
        deck_after_second = box.get_by_uuid(deck_uuid)
        assert cnc_card.nocab_uuid in deck_after_second.card_nocab_uuids
        assert unknown.nocab_uuid not in deck_after_second.card_nocab_uuids

    def test_reordered_but_same_content_card_list_is_not_reported_as_changed(
        self, tmp_path: Path
    ) -> None:
        binder = _fabtcg_card_binder(["Dorinthea Ironsong", "Command and Conquer"])
        box = DeckBox()
        first_dir = tmp_path / "first"
        first_dir.mkdir()
        _write_fragment(
            first_dir,
            "deck-1",
            _fragment([(1, "Dorinthea Ironsong"), (1, "Command and Conquer")]),
        )
        stage = FabtcgDecklistsExtractionStage()
        first = stage.extract(first_dir, box, binder)
        assert len(first) == 1

        # Same slug, same cards, different raw order — must not be
        # reported as a change (card_nocab_uuids is an unordered
        # multiset, see src/schema/card.py).
        reordered_dir = tmp_path / "reordered"
        reordered_dir.mkdir()
        _write_fragment(
            reordered_dir,
            "deck-1",
            _fragment([(1, "Command and Conquer"), (1, "Dorinthea Ironsong")]),
        )

        second = stage.extract(reordered_dir, box, binder)

        assert second == []
        assert len(list(box.all_decks(GameId.FLESH_AND_BLOOD))) == 1

    def test_raises_runtime_error_when_unknown_sentinel_not_seeded(
        self, tmp_path: Path
    ) -> None:
        binder = _fabtcg_card_binder(["Dorinthea Ironsong"])
        box = DeckBox()
        _write_fragment(tmp_path, "deck-1", _fragment([(1, "Nonexistent Card")]))
        stage = FabtcgDecklistsExtractionStage()

        with pytest.raises(RuntimeError):
            stage.extract(tmp_path, box, binder)

    def test_fragment_with_zero_card_items_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        binder = _fabtcg_card_binder([])
        box = DeckBox()
        _write_fragment(tmp_path, "deck-1", _fragment([]))
        stage = FabtcgDecklistsExtractionStage()

        with pytest.raises(ValueError):
            stage.extract(tmp_path, box, binder)
