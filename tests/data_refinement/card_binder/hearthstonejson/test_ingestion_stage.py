import json
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.hearthstonejson.ingestion_stage import (
    HearthstoneJsonCardIngestionStage,
)
from src.schema.card import GenericCard
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _row(dbf_id: int, **fields: object) -> dict:
    base = {
        "dbfId": dbf_id,
        "id": f"ID_{dbf_id}",
        "collectible": True,
        "name": "Holy Smite",
        "cost": 1,
        "type": "SPELL",
        "cardClass": "PRIEST",
        "rarity": "FREE",
        "set": "LEGACY",
        "text": "Deal $3 damage\nto a minion.",
        "artist": "Someone",
        "flavor": "Smite!",
        "mechanics": ["TRIGGER_VISUAL"],
    }
    return {**base, **fields}


def _ingest(rows: list[dict], tmp_path: Path, binder: CardBinder | None = None):
    raw = tmp_path / "123.json"
    raw.write_text(json.dumps(rows), encoding="utf-8")
    binder = binder if binder is not None else CardBinder()
    changed = HearthstoneJsonCardIngestionStage().ingest(raw, binder)
    return binder, changed


class TestRowsThatAreCards:
    def test_keeps_collectible_non_cosmetic_rows_only(self, tmp_path: Path) -> None:
        rows = [
            _row(1),
            _row(2, name="Token", collectible=False),
            _row(3, name="Skin", set="HERO_SKINS", type="HERO"),
        ]

        binder, _ = _ingest(rows, tmp_path)

        assert [c.name for c in binder.all_cards(GameId.HEARTHSTONE)] == ["Holy Smite"]


class TestEmptyBuild:
    def test_a_build_without_deck_cards_raises_and_keeps_the_binder(
        self, tmp_path: Path
    ) -> None:
        binder, _ = _ingest([_row(1)], tmp_path)

        with pytest.raises(ValueError):
            _ingest([_row(2, collectible=False)], tmp_path, binder)
        assert len(list(binder.all_cards(GameId.HEARTHSTONE))) == 1


class TestIdentity:
    def test_reprints_collapse_and_every_dbf_id_is_an_alias(
        self, tmp_path: Path
    ) -> None:
        rows = [_row(900, set="CORE", rarity="COMMON"), _row(5, set="EXPERT1")]

        binder, changed = _ingest(rows, tmp_path)

        (card,) = binder.all_cards(GameId.HEARTHSTONE)
        assert changed == [card.nocab_uuid]
        assert card.raw_content["set"] == "EXPERT1"  # lowest dbfId wins
        assert card.provenance.source_id == "5"
        for dbf_id in ("5", "900"):
            found = binder.get_by_alias(
                GameId.HEARTHSTONE, DataSource.HEARTHSTONEJSON, dbf_id
            )
            assert found is not None and found.nocab_uuid == card.nocab_uuid

    def test_a_rebalanced_version_stays_its_own_card(self, tmp_path: Path) -> None:
        rows = [_row(1, set="VANILLA", cost=5), _row(2, set="LEGACY", cost=3)]

        binder, _ = _ingest(rows, tmp_path)

        costs = sorted(
            c.raw_content["cost"] for c in binder.all_cards(GameId.HEARTHSTONE)
        )
        assert costs == [3, 5]

    def test_reingesting_the_same_build_changes_nothing(self, tmp_path: Path) -> None:
        binder, _ = _ingest([_row(1)], tmp_path)
        uuid_before = next(iter(binder.all_cards(GameId.HEARTHSTONE))).nocab_uuid

        binder, changed = _ingest([_row(1)], tmp_path, binder)

        assert changed == []
        assert next(iter(binder.all_cards(GameId.HEARTHSTONE))).nocab_uuid == (
            uuid_before
        )


class TestLaterBuilds:
    def _card_for(self, binder: CardBinder, dbf_id: str) -> GenericCard:
        card = binder.get_by_alias(
            GameId.HEARTHSTONE, DataSource.HEARTHSTONEJSON, dbf_id
        )
        assert card is not None
        return card

    def test_a_rebalance_that_splits_a_group_keeps_both_cards(
        self, tmp_path: Path
    ) -> None:
        binder, _ = _ingest([_row(5), _row(900, set="CORE")], tmp_path)
        original = self._card_for(binder, "5").nocab_uuid

        binder, _ = _ingest([_row(5), _row(900, set="CORE", cost=2)], tmp_path, binder)

        assert self._card_for(binder, "5").nocab_uuid == original
        assert self._card_for(binder, "5").raw_content["cost"] == 1
        assert self._card_for(binder, "900").raw_content["cost"] == 2
        assert len(list(binder.all_cards(GameId.HEARTHSTONE))) == 2

    def test_a_merge_leaves_no_stale_card_behind(self, tmp_path: Path) -> None:
        binder, _ = _ingest([_row(1, cost=5), _row(2)], tmp_path)

        binder, _ = _ingest([_row(1), _row(2)], tmp_path, binder)

        (card,) = binder.all_cards(GameId.HEARTHSTONE)
        assert self._card_for(binder, "2").nocab_uuid == card.nocab_uuid

    def test_cards_missing_from_the_new_build_are_deleted_but_not_unknown(
        self, tmp_path: Path
    ) -> None:
        binder, _ = _ingest([_row(1), _row(2, name="Gone")], tmp_path)
        unknown = binder.ensure_unknown_card(GameId.HEARTHSTONE)

        binder, _ = _ingest([_row(1)], tmp_path, binder)

        names = sorted(c.name for c in binder.all_cards(GameId.HEARTHSTONE))
        assert names == sorted(["Holy Smite", unknown.name])


class TestLeanContent:
    def test_keeps_game_fields_in_order_and_cleans_text(self, tmp_path: Path) -> None:
        binder, _ = _ingest([_row(1)], tmp_path)

        (card,) = binder.all_cards(GameId.HEARTHSTONE)
        assert list(card.raw_content.items()) == [
            ("name", "Holy Smite"),
            ("cost", 1),
            ("type", "SPELL"),
            ("cardClass", "PRIEST"),
            ("rarity", "FREE"),
            ("set", "LEGACY"),
            ("text", "Deal 3 damage to a minion."),
        ]

    def test_drops_placeholders_and_progress_text(self, tmp_path: Path) -> None:
        text = "Battlecry: Summon a{1} {0} Jade Golem.@ ({0} left!)@ (Ready!)"
        binder, _ = _ingest([_row(1, text=text)], tmp_path)

        (card,) = binder.all_cards(GameId.HEARTHSTONE)
        assert card.raw_content["text"] == "Battlecry: Summon a Jade Golem."

    def test_drops_zero_durability_and_zero_runes_and_markup(
        self, tmp_path: Path
    ) -> None:
        rows = [
            _row(
                1,
                name="Blade",
                type="WEAPON",
                attack=2,
                health=3,
                durability=0,
                runeCost={"blood": 2, "frost": 0, "unholy": 0},
                text="[x]<b>Battlecry:</b> Gain <i>#4</i> Armor.",
            )
        ]

        binder, _ = _ingest(rows, tmp_path)

        (card,) = binder.all_cards(GameId.HEARTHSTONE)
        assert "durability" not in card.raw_content
        assert card.raw_content["runeCost"] == {"blood": 2}
        assert card.raw_content["text"] == "Battlecry: Gain 4 Armor."
