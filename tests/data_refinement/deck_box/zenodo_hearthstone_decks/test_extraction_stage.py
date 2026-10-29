import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.zenodo_hearthstone_decks.extraction_stage import (
    CardNameMatcher,
    ZenodoHearthstoneDecksExtractionStage,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_DECKS_HEADER = (
    "idDeck;Minions;Spells;Weapons;DeckType;DeckArchetype;CraftingCost;Date;Set\n"
)


def _add_card(binder: CardBinder, name: str, dbf_id: int, card_set: str) -> UUID:
    card = GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.HEARTHSTONE,
        name=name,
        raw_content={"name": name, "set": card_set},
        provenance=Provenance(
            data_source=DataSource.HEARTHSTONEJSON,
            source_id=str(dbf_id),
            fetched_at=datetime.now(timezone.utc),
        ),
    )
    binder.create(card)
    return card.nocab_uuid


def _binder() -> tuple[CardBinder, dict[str, UUID]]:
    binder = CardBinder()
    uuids = {
        "Fireball": _add_card(binder, "Fireball", 315, "EXPERT1"),
        "Frostbolt": _add_card(binder, "Frostbolt", 662, "LEGACY"),
    }
    binder.ensure_unknown_card(GameId.HEARTHSTONE)
    return binder, uuids


def _write_raw(
    tmp_path: Path, card_rows: list[tuple[str, str]], deck_rows: list[str]
) -> Path:
    (tmp_path / "cards.csv").write_text(
        "idDeck;idCard\n" + "".join(f"{d};{n}\n" for d, n in card_rows),
        encoding="utf-8",
    )
    (tmp_path / "decks.csv").write_text(
        _DECKS_HEADER + "".join(f"{row}\n" for row in deck_rows), encoding="utf-8"
    )
    return tmp_path


def _deck_row(deck_id: str, archetype: str) -> str:
    return f"{deck_id};10;20;0;Ranked Deck;{archetype};5000;12/11/2017;Kobolds Patch"


class TestExtract:
    def test_stores_a_deck_as_a_copy_count_multiset(self, tmp_path: Path) -> None:
        binder, uuids = _binder()
        cards = [("7", "Fireball ")] * 2 + [("7", "Frostbolt")] * 28
        raw_dir = _write_raw(tmp_path, cards, [_deck_row("7", "Freeze Mage")])
        box = DeckBox()

        created = ZenodoHearthstoneDecksExtractionStage().extract(raw_dir, box, binder)

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.name == "Freeze Mage"
        assert Counter(deck.card_nocab_uuids) == {
            uuids["Fireball"]: 2,
            uuids["Frostbolt"]: 28,
        }
        assert deck.provenance is not None
        assert deck.provenance.data_source == DataSource.ZENODO_HEARTHSTONE_DECKS
        assert deck.provenance.source_id == "7"

    def test_names_a_deck_by_id_without_a_known_archetype(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        cards = [("7", "Fireball"), ("8", "Frostbolt")]
        deck_rows = [
            _deck_row("7", "Unknown"),
            "1;2;" + _deck_row("8", "Face Hunter"),  # malformed: skipped
        ]
        box = DeckBox()

        created = ZenodoHearthstoneDecksExtractionStage().extract(
            _write_raw(tmp_path, cards, deck_rows), box, binder
        )

        decks = [box.get_by_uuid(nocab_uuid) for nocab_uuid in created]
        assert sorted(deck.name for deck in decks if deck) == ["7", "8"]

    def test_identical_lists_are_one_deck_and_rerun_creates_nothing(
        self, tmp_path: Path
    ) -> None:
        binder, _ = _binder()
        raw_dir = _write_raw(tmp_path, [("7", "Fireball"), ("8", "Fireball")], [])
        box = DeckBox()
        stage = ZenodoHearthstoneDecksExtractionStage()

        first = stage.extract(raw_dir, box, binder)
        second = stage.extract(raw_dir, box, binder)

        assert len(first) == 1 and second == []
        assert list(box.all_uuids(GameId.HEARTHSTONE)) == first

    def test_an_unknown_name_maps_to_the_unknown_card(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder, uuids = _binder()
        cards = [("7", "Fireball"), ("7", "Mistress of Pain")]
        box = DeckBox()

        with caplog.at_level(logging.WARNING):
            created = ZenodoHearthstoneDecksExtractionStage().extract(
                _write_raw(tmp_path, cards, []), box, binder
            )

        deck = box.get_by_uuid(created[0])
        assert deck is not None
        unknown = binder.get_by_name_single(GameId.HEARTHSTONE, "Unknown")
        assert unknown is not None
        assert Counter(deck.card_nocab_uuids) == {
            uuids["Fireball"]: 1,
            unknown.nocab_uuid: 1,
        }
        assert "Mistress of Pain" in caplog.text

    def test_a_card_named_unknown_maps_to_the_sentinel_without_crashing(
        self, tmp_path: Path
    ) -> None:
        binder, _ = _binder()
        box = DeckBox()

        created = ZenodoHearthstoneDecksExtractionStage().extract(
            _write_raw(tmp_path, [("7", "Unknown")], []), box, binder
        )

        deck = box.get_by_uuid(created[0])
        unknown = binder.get_by_name_single(GameId.HEARTHSTONE, "Unknown")
        assert deck is not None and unknown is not None
        assert deck.card_nocab_uuids == [unknown.nocab_uuid]

    def test_non_contiguous_deck_rows_raise(self, tmp_path: Path) -> None:
        binder, _ = _binder()
        cards = [("7", "Fireball"), ("8", "Fireball"), ("7", "Frostbolt")]

        with pytest.raises(ValueError, match="contiguous"):
            ZenodoHearthstoneDecksExtractionStage().extract(
                _write_raw(tmp_path, cards, []), DeckBox(), binder
            )

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        binder, _ = _binder()

        with pytest.raises(FileNotFoundError):
            ZenodoHearthstoneDecksExtractionStage().extract(tmp_path, DeckBox(), binder)


class TestCardNameMatcher:
    def test_takes_the_newest_non_classic_version(self) -> None:
        binder = CardBinder()
        _add_card(binder, "Holy Nova", 68299, "VANILLA")
        legacy = _add_card(binder, "Holy Nova", 841, "LEGACY")
        _add_card(binder, "Spectral Pillager", 45975, "ICECROWN")
        core = _add_card(binder, "Spectral Pillager", 95757, "CORE_HIDDEN")
        matcher = CardNameMatcher(binder, uuid4())

        result = matcher.match_card_nocab_uuids(("Holy Nova", "Spectral Pillager"))

        assert result == [legacy, core]
        assert not matcher.unmatched_counts

    def test_a_classic_only_card_still_matches(self) -> None:
        binder = CardBinder()
        vanilla = _add_card(binder, "Old Card", 68000, "VANILLA")

        result = CardNameMatcher(binder, uuid4()).match_card_nocab_uuids(("Old Card",))

        assert result == [vanilla]
