import json
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.dominiontabs.ingestion_stage import (
    DominionTabsCardIngestionStage,
    _clean_description,
)
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_MILITIA_DB = {
    "card_tag": "Militia",
    "cardset_tags": ["dominion1stEdition", "dominion2ndEdition"],
    "cost": "4",
    "types": ["Action", "Attack"],
}
_MILITIA_EN = {
    "description": "+2 Coins<n>Each other player discards down to 3 cards.",
    "extra": "A long ruling that is not on the card.",
    "name": "Militia",
}


def _write_raw(path: Path, db: list[dict], en: dict[str, dict]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "cards_db.json").write_text(json.dumps(db))
    (path / "cards_en_us.json").write_text(json.dumps(en))
    return path


def _ingest_one(tmp_path: Path, db_entry: dict, en_entry: dict) -> GenericCard:
    en = {db_entry["card_tag"]: en_entry}
    raw_dir = _write_raw(tmp_path, [db_entry], en)
    binder = CardBinder()
    DominionTabsCardIngestionStage().ingest(raw_dir, binder)
    card = binder.get_by_alias(
        GameId.DOMINION, DataSource.DOMINIONTABS, db_entry["card_tag"]
    )
    assert card is not None
    return card


def _card(binder: CardBinder, card_tag: str) -> GenericCard:
    card = binder.get_by_alias(GameId.DOMINION, DataSource.DOMINIONTABS, card_tag)
    assert card is not None
    return card


class TestIngest:
    def test_joins_the_two_files_by_card_tag(self, tmp_path: Path) -> None:
        card = _ingest_one(tmp_path, _MILITIA_DB, _MILITIA_EN)

        assert card.name == "Militia"
        assert card.source_game == GameId.DOMINION
        assert card.provenance.source_id == "Militia"
        assert card.raw_content == {
            "name": "Militia",
            "types": ["Action", "Attack"],
            "cost": "4",
            "description": "+2 Coins\nEach other player discards down to 3 cards.",
        }

    def test_other_keys_and_non_card_english_entries_are_left_out(
        self, tmp_path: Path
    ) -> None:
        db = [
            {
                **_MILITIA_DB,
                "group_tag": "x",
                "group_top": True,
                "randomizer": False,
                "count": "4",
                "image": "militia.png",
            }
        ]
        header = {"description": "A header.", "name": "Boons"}
        en = {"Militia": _MILITIA_EN, "Boons": header}
        binder = CardBinder()

        DominionTabsCardIngestionStage().ingest(_write_raw(tmp_path, db, en), binder)

        cards = list(binder.all_cards(GameId.DOMINION))
        assert [c.name for c in cards] == ["Militia"]
        text = json.dumps(cards[0].raw_content)
        for left_out in ("cardset", "group", "randomizer", "count", "png", "ruling"):
            assert left_out not in text

    def test_potcost_and_debtcost_are_kept_only_when_present(
        self, tmp_path: Path
    ) -> None:
        alchemist = _ingest_one(
            tmp_path / "a",
            {"card_tag": "Alchemist", "types": ["Action"], "cost": "3", "potcost": "1"},
            {"description": "+2 Cards", "name": "Alchemist"},
        )
        engineer = _ingest_one(
            tmp_path / "b",
            {"card_tag": "Engineer", "types": ["Action"], "cost": "0", "debtcost": "4"},
            {"description": "Gain a card.", "name": "Engineer"},
        )
        militia = _ingest_one(tmp_path / "c", _MILITIA_DB, _MILITIA_EN)

        assert alchemist.raw_content["potcost"] == "1"
        assert engineer.raw_content["debtcost"] == "4"
        assert "potcost" not in militia.raw_content
        assert "debtcost" not in militia.raw_content

    def test_cost_is_always_present_even_when_the_card_has_none(
        self, tmp_path: Path
    ) -> None:
        # CostRegressionMetric indexes raw_content["cost"] directly and reads
        # "" together with a potcost as a real cost of 0 (Transmute).
        blank = _ingest_one(
            tmp_path / "a",
            {"card_tag": "Transmute", "types": ["Action"], "cost": "", "potcost": "1"},
            {"description": "Trash a card.", "name": "Transmute"},
        )
        absent = _ingest_one(
            tmp_path / "b",
            {"card_tag": "Some Prophecy", "types": ["Prophecy"]},
            {"description": "Something.", "name": "Some Prophecy"},
        )

        assert blank.raw_content["cost"] == ""
        assert absent.raw_content["cost"] == ""

    def test_costs_keep_their_raw_suffix_strings(self, tmp_path: Path) -> None:
        card = _ingest_one(
            tmp_path,
            {"card_tag": "Peddler", "types": ["Action"], "cost": "8*"},
            {"description": "+1 Card", "name": "Peddler"},
        )

        assert card.raw_content["cost"] == "8*"

    def test_entry_without_english_text_is_skipped(self, tmp_path: Path) -> None:
        # The two German big-box duplicates ("Soothsayer BB2DE") have no
        # English text and would otherwise become junk-named cards.
        db = [
            _MILITIA_DB,
            {"card_tag": "Soothsayer BB2DE", "types": ["Action"], "cost": "5"},
        ]
        binder = CardBinder()

        changed = DominionTabsCardIngestionStage().ingest(
            _write_raw(tmp_path, db, {"Militia": _MILITIA_EN}), binder
        )

        assert [c.name for c in binder.all_cards(GameId.DOMINION)] == ["Militia"]
        assert len(changed) == 1
        assert (
            binder.get_by_alias(
                GameId.DOMINION, DataSource.DOMINIONTABS, "Soothsayer BB2DE"
            )
            is None
        )

    def test_card_whose_text_cleans_to_nothing_has_no_description_key(
        self, tmp_path: Path
    ) -> None:
        card = _ingest_one(
            tmp_path,
            {"card_tag": "Blank", "types": ["Action"], "cost": "1"},
            {"description": "<br>  ", "name": "Blank"},
        )

        assert "description" not in card.raw_content

    def test_types_keep_their_order(self, tmp_path: Path) -> None:
        card = _ingest_one(
            tmp_path,
            {
                "card_tag": "Black Cat",
                "types": ["Action", "Attack", "Reaction"],
                "cost": "2",
            },
            {"description": "+2 Cards", "name": "Black Cat"},
        )

        assert card.raw_content["types"] == ["Action", "Attack", "Reaction"]

    def test_key_order_puts_description_last(self, tmp_path: Path) -> None:
        card = _ingest_one(
            tmp_path,
            {"card_tag": "Alchemist", "types": ["Action"], "cost": "3", "potcost": "1"},
            {"description": "+2 Cards", "name": "Alchemist"},
        )

        assert list(card.raw_content) == [
            "name",
            "types",
            "cost",
            "potcost",
            "description",
        ]

    def test_each_entry_becomes_its_own_card(self, tmp_path: Path) -> None:
        db = [_MILITIA_DB, {"card_tag": "Moat", "types": ["Action"], "cost": "2"}]
        moat = {"description": "+2 Cards", "name": "Moat"}
        binder = CardBinder()

        changed = DominionTabsCardIngestionStage().ingest(
            _write_raw(tmp_path, db, {"Militia": _MILITIA_EN, "Moat": moat}), binder
        )

        assert len(changed) == 2
        assert set(binder.all_uuids(GameId.DOMINION)) == set(changed)

    def test_raises_if_a_raw_file_is_missing(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            DominionTabsCardIngestionStage().ingest(tmp_path, CardBinder())

    def test_raises_if_an_entry_has_no_types(self, tmp_path: Path) -> None:
        broken_text = {"description": "x", "name": "Broken"}
        raw_dir = _write_raw(
            tmp_path, [{"card_tag": "Broken"}], {"Broken": broken_text}
        )

        with pytest.raises(KeyError):
            DominionTabsCardIngestionStage().ingest(raw_dir, CardBinder())


class TestReIngest:
    def test_unchanged_entry_is_a_noop_and_keeps_uuid(self, tmp_path: Path) -> None:
        stage, binder = DominionTabsCardIngestionStage(), CardBinder()
        first = _write_raw(tmp_path / "1", [_MILITIA_DB], {"Militia": _MILITIA_EN})
        stage.ingest(first, binder)
        original = _card(binder, "Militia")

        second = _write_raw(tmp_path / "2", [_MILITIA_DB], {"Militia": _MILITIA_EN})
        changed = stage.ingest(second, binder)

        assert changed == []
        assert _card(binder, "Militia") == original

    def test_changed_text_replaces_content_and_keeps_uuid_even_if_shorter(
        self, tmp_path: Path
    ) -> None:
        stage, binder = DominionTabsCardIngestionStage(), CardBinder()
        first = _write_raw(tmp_path / "1", [_MILITIA_DB], {"Militia": _MILITIA_EN})
        stage.ingest(first, binder)
        original = _card(binder, "Militia")
        shorter = {"description": "+2 Coins", "name": "Militia"}

        second = _write_raw(tmp_path / "2", [_MILITIA_DB], {"Militia": shorter})
        changed = stage.ingest(second, binder)

        updated = _card(binder, "Militia")
        assert updated.nocab_uuid == original.nocab_uuid
        assert updated.raw_content["description"] == "+2 Coins"
        assert changed == [original.nocab_uuid]

    def test_change_only_in_dropped_fields_is_a_noop(self, tmp_path: Path) -> None:
        stage, binder = DominionTabsCardIngestionStage(), CardBinder()
        first = _write_raw(tmp_path / "1", [_MILITIA_DB], {"Militia": _MILITIA_EN})
        stage.ingest(first, binder)
        db = [{**_MILITIA_DB, "cardset_tags": ["promo"], "count": "9"}]
        en = {"Militia": {**_MILITIA_EN, "extra": "A different ruling."}}

        changed = stage.ingest(_write_raw(tmp_path / "2", db, en), binder)

        assert changed == []


class TestCleanDescription:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("+1 Card<br>+2 Actions", "+1 Card\n+2 Actions"),
            ("+2 Coins<n>Each other player", "+2 Coins\nEach other player"),
            ("+2 Cards<line>When you discard", "+2 Cards\nWhen you discard"),
            ("Reveal.<br><br>Then take", "Reveal.\n\nThen take"),
            ("A<br><br><br><br>B", "A\n\nB"),
            ("1 <*COIN*>", "1 Coin"),
            ("2 <VP>", "2 VP"),
            ("worth <*VP*>", "worth VP"),
            ("costs <*POTION*>", "costs Potion"),
            ("<b>Note</b> and <i>italics</i>", "Note and italics"),
            ("<justify>Events are not cards.</justify>", "Events are not cards."),
            ("  spaced   out text  ", "spaced out text"),
            ("a <br> b", "a\nb"),
            ("plain text", "plain text"),
            ("", ""),
        ],
    )
    def test_cleans_markup(self, raw: str, expected: str) -> None:
        assert _clean_description(raw) == expected

    def test_underlined_names_in_left_blocks_keep_their_separation(self) -> None:
        raw = (
            "Pile:<left>5 Coin<u>Dame Anna</u>: Trash.</left>"
            "<left>4 Coin<u>Sir Martin</u>: +2 Buys</left>"
        )

        assert _clean_description(raw) == (
            "Pile:\n5 Coin Dame Anna: Trash.\n4 Coin Sir Martin: +2 Buys"
        )

    def test_no_angle_bracket_tags_remain_for_known_markup(self) -> None:
        raw = "<center><b>Title</b></center><left>x<u>y</u></left><i>z</i><br><n><line>"

        assert "<" not in _clean_description(raw)
