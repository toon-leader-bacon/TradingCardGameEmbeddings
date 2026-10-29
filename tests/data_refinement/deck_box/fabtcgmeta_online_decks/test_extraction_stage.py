import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.fabtcgmeta_online_decks.extraction_stage import (
    FabtcgmetaOnlineDecksExtractionStage,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _add_card(binder: CardBinder, name: str, source_id: str, pitch: str = "") -> UUID:
    raw_content = {"name": name, "pitch": pitch} if pitch else {"name": name}
    card = GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.FLESH_AND_BLOOD,
        name=name,
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.CARDVAULT_FABTCG,
            source_id=source_id,
            fetched_at=datetime.now(timezone.utc),
        ),
    )
    binder.create(card)
    return card.nocab_uuid


def _binder() -> tuple[CardBinder, dict[str, UUID]]:
    binder = CardBinder()
    uuids = {
        "Briar": _add_card(binder, "Briar", "briar"),
        "Harness Lightning red": _add_card(
            binder, "Harness Lightning", "harness-lightning-1", "1"
        ),
        "Harness Lightning blue": _add_card(
            binder, "Harness Lightning", "harness-lightning-3", "3"
        ),
        "Arcane Seeds": _add_card(binder, "Arcane Seeds||Life", "arcane-seeds-1", "1"),
        "Swordmaster's Path": _add_card(
            binder, "Swordmaster's Path", "swordmasters-path-1", "1"
        ),
    }
    binder.ensure_unknown_card(GameId.FLESH_AND_BLOOD)
    return binder, uuids


def _entry(name: str, count: int, pitch: str) -> dict:
    return {"name": name, "count": count, "pitch": pitch, "image": "", "price": 0.0}


def _deck(deck_hash: str, cards: dict[str, list[dict]]) -> dict:
    return {
        "deckName": deck_hash,
        "deckInfo": {
            "heroFormat": "Briar, Warden of Thorns Silver Age-O",
            "hero": "Briar, Warden of Thorns",
            "format": "Silver Age-O",
            "date": "2026-06-28",
        },
        "cards": cards,
    }


def _briar_cards() -> dict[str, list[dict]]:
    return {
        "heroWeaponEquipment": [_entry("Briar", 1, "0")],
        "redPitchCards": [
            _entry("Harness Lightning", 2, "1"),
            _entry("Arcane Seeds // Life", 2, "1"),
        ],
        "yellowPitchCards": [],
        "bluePitchCards": [_entry("Harness Lightning", 3, "3")],
    }


def _write_shard(raw_dir: Path, tier: str, prefix: str, decks: list[dict]) -> None:
    (raw_dir / tier).mkdir(parents=True, exist_ok=True)
    shard = {deck["deckName"]: deck for deck in decks}
    (raw_dir / tier / f"{prefix}.json").write_text(json.dumps(shard), encoding="utf-8")


class TestExtract:
    def test_stores_every_group_as_a_copy_count_multiset(self, tmp_path: Path) -> None:
        binder, uuids = _binder()
        _write_shard(tmp_path, "archive", "2e6", [_deck("2e6aa", _briar_cards())])
        box = DeckBox()

        created = FabtcgmetaOnlineDecksExtractionStage().extract(tmp_path, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None
        assert deck.name == "Briar, Warden of Thorns Silver Age-O"
        assert Counter(deck.card_nocab_uuids) == {
            uuids["Briar"]: 1,
            uuids["Harness Lightning red"]: 2,
            uuids["Arcane Seeds"]: 2,
            uuids["Harness Lightning blue"]: 3,
        }
        assert deck.provenance is not None
        assert deck.provenance.data_source == DataSource.FABTCGMETA_ONLINE_DECKS
        assert deck.provenance.source_id == "2e6aa"

    def test_matches_case_and_punctuation_differences(self, tmp_path: Path) -> None:
        binder, uuids = _binder()
        cards = {"redPitchCards": [_entry("SWORDMASTERS PATH", 2, "1")]}
        _write_shard(tmp_path, "archive", "000", [_deck("000aa", cards)])
        box = DeckBox()

        created = FabtcgmetaOnlineDecksExtractionStage().extract(tmp_path, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None
        assert deck.card_nocab_uuids == [uuids["Swordmaster's Path"]] * 2

    def test_matches_a_two_faced_card_by_its_front_face(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        two_faced = GenericCard(
            nocab_uuid=uuid4(),
            source_game=GameId.FLESH_AND_BLOOD,
            name="A Drop in the Ocean // Inner Chi",
            raw_content={"name": "A Drop in the Ocean", "pitch": "3"},
            provenance=Provenance(
                data_source=DataSource.CARDVAULT_FABTCG,
                source_id="a-drop-in-the-ocean-3--inner-chi-3",
                fetched_at=datetime.now(timezone.utc),
            ),
        )
        binder.create(two_faced)
        cards = {"bluePitchCards": [_entry("A Drop in the Ocean", 1, "3")]}
        _write_shard(tmp_path, "archive", "000", [_deck("000aa", cards)])
        box = DeckBox()

        created = FabtcgmetaOnlineDecksExtractionStage().extract(tmp_path, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None
        assert deck.card_nocab_uuids == [two_faced.nocab_uuid]

    def test_an_unknown_name_maps_to_the_unknown_card(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder, _ = _binder()
        cards = {"redPitchCards": [_entry("Fire That Burns Within", 1, "1")]}
        _write_shard(tmp_path, "archive", "000", [_deck("000aa", cards)])
        box = DeckBox()

        with caplog.at_level(logging.WARNING):
            created = FabtcgmetaOnlineDecksExtractionStage().extract(
                tmp_path, box, binder
            )

        deck = box.get_by_uuid(created[0])
        assert deck is not None
        assert deck.card_nocab_uuids == [
            CardBinder.unknown_card_uuid(GameId.FLESH_AND_BLOOD)
        ]
        assert "Fire That Burns Within" in caplog.text

    def test_identical_lists_are_one_deck_and_rerun_creates_nothing(
        self, tmp_path: Path
    ) -> None:
        binder, _ = _binder()
        _write_shard(tmp_path, "archive", "000", [_deck("000aa", _briar_cards())])
        _write_shard(tmp_path, "recent", "fff", [_deck("fffaa", _briar_cards())])
        box = DeckBox()
        stage = FabtcgmetaOnlineDecksExtractionStage()

        first = stage.extract(tmp_path, box, binder)
        second = stage.extract(tmp_path, box, binder)

        assert len(first) == 1 and second == []

    def test_skips_a_malformed_deck(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        no_count = {"redPitchCards": [{"name": "Harness Lightning", "pitch": "1"}]}
        zero_count = {"redPitchCards": [_entry("Harness Lightning", 0, "1")]}
        decks = [
            {"deckName": "000aa", "deckInfo": {}},
            _deck("000bb", no_count),
            _deck("000cc", _briar_cards()),
            _deck("000dd", {"redPitchCards": [], "bluePitchCards": []}),
            _deck("000ee", zero_count),
        ]
        _write_shard(tmp_path, "archive", "000", decks)

        created = FabtcgmetaOnlineDecksExtractionStage().extract(
            tmp_path, DeckBox(), binder
        )

        assert len(created) == 1

    def test_no_shards_raises(self, tmp_path: Path) -> None:
        binder, _ = _binder()

        with pytest.raises(FileNotFoundError):
            FabtcgmetaOnlineDecksExtractionStage().extract(tmp_path, DeckBox(), binder)

    def test_a_truncated_shard_raises_naming_it(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        (tmp_path / "archive").mkdir()
        (tmp_path / "archive" / "000.json").write_text('{"000aa": {', encoding="utf-8")

        with pytest.raises(ValueError, match="000.json"):
            FabtcgmetaOnlineDecksExtractionStage().extract(tmp_path, DeckBox(), binder)
