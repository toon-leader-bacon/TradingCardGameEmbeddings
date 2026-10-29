from collections import Counter
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.hearthstonetopdecks.extraction_stage import (
    HearthstoneTopDecksDeckExtractionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.deck_box.hearthstonetopdecks.fixtures import (
    card_uuid,
    deck_row,
    encode_deckstring,
    make_binder,
    write_decks,
)


class TestExtract:
    def test_stores_a_deck_as_a_copy_count_multiset(self, tmp_path: Path) -> None:
        binder = make_binder({"Fireball": [315, 69620], "Frostbolt": [662]})
        code = encode_deckstring({69620: 2, 662: 28})
        box = DeckBox()

        created = HearthstoneTopDecksDeckExtractionStage().extract(
            write_decks(tmp_path, [deck_row("burn-mage", code, "Burn Mage")]),
            box,
            binder,
        )

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.name == "Burn Mage"
        assert Counter(deck.card_nocab_uuids) == {
            card_uuid(binder, 315): 2,
            card_uuid(binder, 662): 28,
        }
        assert deck.provenance is not None
        assert deck.provenance.data_source == DataSource.HEARTHSTONETOPDECKS
        assert deck.provenance.source_id == "burn-mage"

    def test_reprint_dbf_ids_make_the_same_deck(self, tmp_path: Path) -> None:
        binder = make_binder({"Fireball": [315, 69620]})
        rows = [
            deck_row("a", encode_deckstring({315: 30})),
            deck_row("b", encode_deckstring({69620: 30})),
        ]
        box = DeckBox()

        created = HearthstoneTopDecksDeckExtractionStage().extract(
            write_decks(tmp_path, rows), box, binder
        )

        assert len(created) == 1
        assert list(box.all_uuids(GameId.HEARTHSTONE)) == created

    def test_rerun_creates_nothing(self, tmp_path: Path) -> None:
        binder = make_binder({"Fireball": [315]})
        decks = write_decks(tmp_path, [deck_row("a", encode_deckstring({315: 30}))])
        stage = HearthstoneTopDecksDeckExtractionStage()
        box = DeckBox()
        stage.extract(decks, box, binder)

        assert stage.extract(decks, box, binder) == []

    def test_skips_missing_bad_and_empty_codes(self, tmp_path: Path) -> None:
        binder = make_binder({"Fireball": [315]})
        rows = [
            deck_row("none", None),
            deck_row("bad", "not base64!!"),
            deck_row("empty", encode_deckstring({})),
        ]

        created = HearthstoneTopDecksDeckExtractionStage().extract(
            write_decks(tmp_path, rows), DeckBox(), binder
        )

        assert created == []

    def test_unmatched_dbf_ids_become_the_unknown_sentinel(
        self, tmp_path: Path
    ) -> None:
        binder = make_binder({"Fireball": [315]})
        code = encode_deckstring({315: 28, 999999: 2})
        box = DeckBox()

        created = HearthstoneTopDecksDeckExtractionStage().extract(
            write_decks(tmp_path, [deck_row("a", code, title=None)]), box, binder
        )

        deck = box.get_by_uuid(created[0])
        assert deck is not None and deck.name == "a"
        unknown_uuid = CardBinder.unknown_card_uuid(GameId.HEARTHSTONE)
        assert Counter(deck.card_nocab_uuids)[unknown_uuid] == 2

    def test_sideboard_cards_are_not_deck_cards(self, tmp_path: Path) -> None:
        binder = make_binder({"E.T.C.": [90749], "Fireball": [315]})
        code = encode_deckstring(
            {90749: 1, 315: 29}, sideboard_counts={(90749, 315): 1}
        )
        box = DeckBox()

        created = HearthstoneTopDecksDeckExtractionStage().extract(
            write_decks(tmp_path, [deck_row("a", code)]), box, binder
        )

        deck = box.get_by_uuid(created[0])
        assert deck is not None and len(deck.card_nocab_uuids) == 30

    def test_raises_without_the_unknown_sentinel(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Unknown sentinel"):
            HearthstoneTopDecksDeckExtractionStage().extract(
                write_decks(tmp_path, []), DeckBox(), CardBinder()
            )

    def test_raises_on_missing_decks_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            HearthstoneTopDecksDeckExtractionStage().extract(
                tmp_path / "missing.jsonl", DeckBox(), make_binder({})
            )
