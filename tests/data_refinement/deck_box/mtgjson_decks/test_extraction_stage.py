import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.mtgjson_decks.extraction_stage import (
    MtgjsonDeckExtractionStage,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_ORACLE_A = "00000000-0000-0000-0000-00000000000a"
_ORACLE_B = "00000000-0000-0000-0000-00000000000b"
_ORACLE_C = "00000000-0000-0000-0000-00000000000c"


def _make_binder(oracle_ids: dict[str, str], unknown: bool = True) -> CardBinder:
    """An MTG binder holding one card per (oracle id -> name)."""
    binder = CardBinder()
    for oracle_id, name in oracle_ids.items():
        card = GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.MTG,
            name=name,
            raw_content={},
            provenance=Provenance(
                data_source=DataSource.SCRYFALL,
                source_id=oracle_id,
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        binder.create(card)
        binder.register_alias(
            GameId.MTG, DataSource.SCRYFALL, oracle_id, card.nocab_uuid
        )
    if unknown:
        binder.ensure_unknown_card(GameId.MTG)
    return binder


def _uuid(binder: CardBinder, oracle_id: str) -> UUID:
    card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, oracle_id)
    assert card is not None
    return card.nocab_uuid


def _unknown(binder: CardBinder) -> UUID:
    return binder.ensure_unknown_card(GameId.MTG).nocab_uuid


def _row(name: str, count: int, oracle_id: str | None) -> dict[str, Any]:
    identifiers = {} if oracle_id is None else {"scryfallOracleId": oracle_id}
    return {"name": name, "count": count, "identifiers": identifiers}


def _write_deck(
    directory: Path,
    stem: str,
    main: list[dict[str, Any]],
    deck_type: str = "Theme Deck",
    **boards: list[dict[str, Any]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = {"name": stem.split("_")[0], "type": deck_type, "mainBoard": main, **boards}
    (directory / f"{stem}.json").write_text(
        json.dumps({"meta": {}, "data": data}), encoding="utf-8"
    )


class TestExtract:
    def test_stores_main_board_and_commander_as_a_copy_count_multiset(
        self, tmp_path: Path
    ) -> None:
        binder = _make_binder({_ORACLE_A: "A", _ORACLE_B: "B", _ORACLE_C: "C"})
        _write_deck(
            tmp_path,
            "Feast_LRW",
            [_row("A", 3, _ORACLE_A)],
            commander=[_row("B", 1, _ORACLE_B)],
            sideBoard=[_row("C", 2, _ORACLE_C)],
            displayCommander=[_row("B", 1, _ORACLE_B)],
            tokens=[_row("C", 1, _ORACLE_C)],
        )
        box = DeckBox()

        created = MtgjsonDeckExtractionStage().extract(tmp_path, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.name == "Feast"
        assert Counter(deck.card_nocab_uuids) == {
            _uuid(binder, _ORACLE_A): 3,
            _uuid(binder, _ORACLE_B): 1,
        }
        assert deck.provenance is not None
        assert deck.provenance.data_source == DataSource.MTGJSON_DECKS
        assert deck.provenance.source_id == "Feast_LRW"

    def test_a_card_listed_twice_adds_up(self, tmp_path: Path) -> None:
        binder = _make_binder({_ORACLE_A: "A"})
        _write_deck(tmp_path, "D_X", [_row("A", 2, _ORACLE_A), _row("A", 1, _ORACLE_A)])
        box = DeckBox()

        created = MtgjsonDeckExtractionStage().extract(tmp_path, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None
        assert Counter(deck.card_nocab_uuids) == {_uuid(binder, _ORACLE_A): 3}

    @pytest.mark.parametrize("oracle_id", [None, _ORACLE_C])
    def test_a_miss_maps_to_unknown_and_is_logged(
        self, tmp_path: Path, oracle_id: str | None, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _make_binder({_ORACLE_A: "A"})
        _write_deck(
            tmp_path, "D_X", [_row("A", 1, _ORACLE_A), _row("Gone", 2, oracle_id)]
        )
        box = DeckBox()

        with caplog.at_level(logging.WARNING):
            created = MtgjsonDeckExtractionStage().extract(tmp_path, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None
        assert Counter(deck.card_nocab_uuids) == {
            _uuid(binder, _ORACLE_A): 1,
            _unknown(binder): 2,
        }
        assert "('Gone', 2)" in caplog.text

    @pytest.mark.parametrize(
        "deck_type", ["Secret Lair Drop", "MTGO Redemption", "Bundle Land Pack"]
    )
    def test_skips_product_types(self, tmp_path: Path, deck_type: str) -> None:
        binder = _make_binder({_ORACLE_A: "A"})
        _write_deck(tmp_path, "D_X", [_row("A", 1, _ORACLE_A)], deck_type=deck_type)

        assert MtgjsonDeckExtractionStage().extract(tmp_path, DeckBox(), binder) == []

    def test_skips_a_deck_with_no_played_cards(self, tmp_path: Path) -> None:
        binder = _make_binder({_ORACLE_A: "A"})
        _write_deck(tmp_path, "D_X", [], sideBoard=[_row("A", 1, _ORACLE_A)])

        assert MtgjsonDeckExtractionStage().extract(tmp_path, DeckBox(), binder) == []

    def test_a_reprinted_list_is_stored_once_and_a_rerun_creates_nothing(
        self, tmp_path: Path
    ) -> None:
        binder = _make_binder({_ORACLE_A: "A"})
        _write_deck(tmp_path, "D_X", [_row("A", 4, _ORACLE_A)])
        _write_deck(tmp_path, "D_Y", [_row("A", 4, _ORACLE_A)])
        stage = MtgjsonDeckExtractionStage()
        box = DeckBox()

        created = stage.extract(tmp_path, box, binder)

        assert len(created) == 1
        assert list(box.all_uuids(GameId.MTG)) == created
        assert stage.extract(tmp_path, box, binder) == []

    def test_ignores_non_json_files(self, tmp_path: Path) -> None:
        binder = _make_binder({_ORACLE_A: "A"})
        _write_deck(tmp_path, "D_X", [_row("A", 1, _ORACLE_A)])
        (tmp_path / "README.txt").write_text("not a deck", encoding="utf-8")

        assert (
            len(MtgjsonDeckExtractionStage().extract(tmp_path, DeckBox(), binder)) == 1
        )

    def test_without_the_unknown_card_raises_before_reading(
        self, tmp_path: Path
    ) -> None:
        binder = _make_binder({_ORACLE_A: "A"}, unknown=False)
        (tmp_path / "Bad_X.json").write_text("not json", encoding="utf-8")

        with pytest.raises(RuntimeError):
            MtgjsonDeckExtractionStage().extract(tmp_path, DeckBox(), binder)

    def test_a_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MtgjsonDeckExtractionStage().extract(
                tmp_path / "absent", DeckBox(), _make_binder({})
            )

    def test_none_reads_the_latest_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binder = _make_binder({_ORACLE_A: "A"})
        decks = tmp_path / "5.3.0+20261105" / "decks"
        _write_deck(decks, "D_X", [_row("A", 1, _ORACLE_A)])
        (tmp_path / "5.3.0+20261105" / "Meta.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            "src.data_retrieval.mtgjson_decks.downloader."
            "MtgjsonDeckFilesDownloader.DEFAULT_RAW_DATA_DIR",
            tmp_path,
        )

        assert len(MtgjsonDeckExtractionStage().extract(None, DeckBox(), binder)) == 1

    @pytest.mark.parametrize("count", [0, -1, 1.5, True, "2"])
    def test_a_bad_count_raises(self, tmp_path: Path, count: Any) -> None:
        binder = _make_binder({_ORACLE_A: "A"})
        _write_deck(tmp_path, "D_X", [{**_row("A", 1, _ORACLE_A), "count": count}])

        with pytest.raises(ValueError, match="positive int"):
            MtgjsonDeckExtractionStage().extract(tmp_path, DeckBox(), binder)

    @pytest.mark.parametrize("missing", ["name", "type"])
    def test_a_deck_missing_a_field_raises(self, tmp_path: Path, missing: str) -> None:
        data = {"name": "D", "type": "Theme Deck", "mainBoard": []}
        del data[missing]
        (tmp_path / "D_X.json").write_text(json.dumps({"data": data}), encoding="utf-8")

        with pytest.raises(ValueError, match="D_X.json"):
            MtgjsonDeckExtractionStage().extract(tmp_path, DeckBox(), _make_binder({}))

    @pytest.mark.parametrize("body", ['{"meta": {}}', "not json"])
    def test_a_file_without_data_raises_naming_it(
        self, tmp_path: Path, body: str
    ) -> None:
        (tmp_path / "D_X.json").write_text(body, encoding="utf-8")

        with pytest.raises(ValueError, match="D_X.json"):
            MtgjsonDeckExtractionStage().extract(tmp_path, DeckBox(), _make_binder({}))

    def test_null_boards_and_identifiers_read_as_empty(self, tmp_path: Path) -> None:
        binder = _make_binder({})
        row = {"name": "A", "count": 1, "identifiers": None}
        _write_deck(tmp_path, "D_X", [row], commander=None)  # type: ignore[arg-type]
        box = DeckBox()

        created = MtgjsonDeckExtractionStage().extract(tmp_path, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.card_nocab_uuids == [_unknown(binder)]
