import json
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.spire_codex.ingestion_stage import (
    SpireCodexCardIngestionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_REAL_ROW = {
    "id": "ABRASIVE",
    "name": "Abrasive",
    "description": "Gain 1 [gold]Dexterity[/gold].\nGain 4 [gold]Thorns[/gold].",
    "cost": 3,
    "type": "Power",
    "rarity": "Rare",
    "color": "silent",
}


def _write_cards_file(path: Path, rows: list[dict]) -> Path:
    cards_path = path / "cards.json"
    cards_path.write_text(json.dumps(rows))
    return cards_path


class TestIngest:
    def test_creates_one_card_per_row(self, tmp_path: Path) -> None:
        cards_path = _write_cards_file(
            tmp_path,
            [
                {"id": "ABRASIVE", "name": "Abrasive", "color": "silent"},
                {"id": "ACCELERANT", "name": "Accelerant", "color": "silent"},
            ],
        )
        binder = CardBinder()

        changed = SpireCodexCardIngestionStage().ingest(cards_path, binder)

        assert len(changed) == 2
        assert set(binder.all_uuids(GameId.SLAY_THE_SPIRE_2)) == set(changed)

    def test_maps_id_to_provenance_source_id(self, tmp_path: Path) -> None:
        cards_path = _write_cards_file(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        SpireCodexCardIngestionStage().ingest(cards_path, binder)

        card = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ABRASIVE"
        )
        assert card is not None
        assert card.provenance.source_id == _REAL_ROW["id"]
        assert card.provenance.data_source == DataSource.SPIRE_CODEX

    def test_raw_content_is_lean_with_color_tags_stripped(self, tmp_path: Path) -> None:
        cards_path = _write_cards_file(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        SpireCodexCardIngestionStage().ingest(cards_path, binder)

        card = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ABRASIVE"
        )
        assert card.raw_content == {
            "name": "Abrasive",
            "cost": 3,
            "type": "Power",
            "rarity": "Rare",
            "color": "silent",
            "description": "Gain 1 Dexterity.\nGain 4 Thorns.",
        }

    def test_name_is_kept_plain_no_disambiguation(self, tmp_path: Path) -> None:
        # Under the v2 design, identity is keyed on "id" (already
        # per-character-unique), not name — so name no longer needs
        # to be mangled to survive a name-uniqueness invariant that no
        # longer exists.
        cards_path = _write_cards_file(tmp_path, [_REAL_ROW])
        binder = CardBinder()

        SpireCodexCardIngestionStage().ingest(cards_path, binder)

        card = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ABRASIVE"
        )
        assert card.name == "Abrasive"

    def test_each_card_gets_a_distinct_uuid(self, tmp_path: Path) -> None:
        cards_path = _write_cards_file(
            tmp_path,
            [
                {"id": "ABRASIVE", "name": "Abrasive", "color": "silent"},
                {"id": "ACCELERANT", "name": "Accelerant", "color": "silent"},
            ],
        )
        binder = CardBinder()

        SpireCodexCardIngestionStage().ingest(cards_path, binder)

        abrasive = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ABRASIVE"
        )
        accelerant = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ACCELERANT"
        )
        assert abrasive.nocab_uuid != accelerant.nocab_uuid

    def test_strike_and_defend_all_become_distinct_cards_by_plain_name(
        self, tmp_path: Path
    ) -> None:
        # The whole point of keying on "id": these 5 characters' worth
        # of Strike/Defend all survive as distinct cards, and
        # get_by_name("Strike") correctly returns every one of them —
        # no _disambiguated_name() workaround needed anymore.
        cards_path = _write_cards_file(
            tmp_path,
            [
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "color": "ironclad"},
                {"id": "STRIKE_SILENT", "name": "Strike", "color": "silent"},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "color": "ironclad"},
            ],
        )
        binder = CardBinder()

        changed = SpireCodexCardIngestionStage().ingest(cards_path, binder)

        assert len(changed) == 3
        strikes = binder.get_by_name(GameId.SLAY_THE_SPIRE_2, "Strike")
        assert len(strikes) == 2
        assert {c.provenance.source_id for c in strikes} == {
            "STRIKE_IRONCLAD",
            "STRIKE_SILENT",
        }
        assert {c.raw_content["color"] for c in strikes} == {"ironclad", "silent"}

    def test_raises_if_raw_path_does_not_exist(self, tmp_path: Path) -> None:
        binder = CardBinder()
        with pytest.raises(OSError):
            SpireCodexCardIngestionStage().ingest(tmp_path / "missing.json", binder)


class TestReIngestDuplicates:
    def test_unchanged_row_is_a_noop_and_preserves_uuid(self, tmp_path: Path) -> None:
        binder = CardBinder()
        stage = SpireCodexCardIngestionStage()
        cards_path = _write_cards_file(tmp_path, [_REAL_ROW])
        stage.ingest(cards_path, binder)
        original = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ABRASIVE"
        )

        (tmp_path / "second").mkdir()
        second_path = _write_cards_file(tmp_path / "second", [_REAL_ROW])
        changed = stage.ingest(second_path, binder)

        reloaded = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ABRASIVE"
        )
        assert reloaded.nocab_uuid == original.nocab_uuid
        assert changed == []

    def test_changed_row_updates_content_in_place_keeping_uuid(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        stage = SpireCodexCardIngestionStage()
        sparse_row = {"id": "ABRASIVE", "name": "Abrasive", "color": "silent"}
        (tmp_path / "sparse").mkdir()
        sparse_path = _write_cards_file(tmp_path / "sparse", [sparse_row])
        stage.ingest(sparse_path, binder)
        original = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ABRASIVE"
        )

        (tmp_path / "richer").mkdir()
        richer_path = _write_cards_file(tmp_path / "richer", [_REAL_ROW])
        changed = stage.ingest(richer_path, binder)

        updated = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ABRASIVE"
        )
        assert updated.nocab_uuid == original.nocab_uuid
        assert updated.raw_content["cost"] == 3
        assert changed == [original.nocab_uuid]

    def test_shorter_incoming_row_still_wins(self, tmp_path: Path) -> None:
        binder = CardBinder()
        stage = SpireCodexCardIngestionStage()
        (tmp_path / "old").mkdir()
        stage.ingest(_write_cards_file(tmp_path / "old", [_REAL_ROW]), binder)

        (tmp_path / "new").mkdir()
        shorter = {"id": "ABRASIVE", "name": "Abrasive", "description": "Gain 1."}
        changed = stage.ingest(_write_cards_file(tmp_path / "new", [shorter]), binder)

        card = binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "ABRASIVE"
        )
        assert card.raw_content == {"name": "Abrasive", "description": "Gain 1."}
        assert changed == [card.nocab_uuid]


class TestLeanContent:
    def _content(self, tmp_path: Path, row: dict) -> dict:
        tmp_path.mkdir(parents=True, exist_ok=True)
        cards_path = _write_cards_file(tmp_path, [{"id": "X", "name": "Card", **row}])
        binder = CardBinder()
        SpireCodexCardIngestionStage().ingest(cards_path, binder)
        card = binder.get_by_alias(GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "X")
        return card.raw_content

    def test_nulls_ids_urls_and_redundant_keys_are_dropped(
        self, tmp_path: Path
    ) -> None:
        content = self._content(
            tmp_path,
            {
                "cost": 1,
                "damage": 6,
                "block": None,
                "description": "Deal 6 damage.",
                "description_raw": "Deal {Damage:diff()} damage.",
                "vars": {"Damage": 6},
                "powers_applied": [{"power": "Weak", "amount": 1}],
                "type": "Attack",
                "type_key": "Attack",
                "rarity": "Basic",
                "rarity_key": "Basic",
                "keywords": ["Exhaust"],
                "keywords_key": ["Exhaust"],
                "compendium_order": 2,
                "image_url": "/static/images/cards/x.webp",
                "beta_image_url": None,
                "spawns_cards": ["SHIV"],
                "sources": [{"type": "monster", "id": "M", "name": "Monster"}],
            },
        )

        assert content == {
            "name": "Card",
            "cost": 1,
            "type": "Attack",
            "rarity": "Basic",
            "keywords": ["Exhaust"],
            "damage": 6,
            "description": "Deal 6 damage.",
        }

    def test_energy_symbols_are_kept_but_color_tags_are_stripped(
        self, tmp_path: Path
    ) -> None:
        content = self._content(
            tmp_path,
            {"description": "Gain [energy:2]. Apply [blue]2[/blue] [gold]Weak[/gold]."},
        )

        assert content["description"] == "Gain [energy:2]. Apply 2 Weak."

    def test_upgrade_is_kept_only_when_it_has_no_rendered_description(
        self, tmp_path: Path
    ) -> None:
        with_text = self._content(
            tmp_path / "a",
            {"upgrade": {"damage": "+3"}, "upgrade_description": "Deal 9 damage."},
        )
        without_text = self._content(tmp_path / "b", {"upgrade": {"add_innate": True}})

        assert "upgrade" not in with_text
        assert with_text["upgrade_description"] == "Deal 9 damage."
        assert without_text["upgrade"] == {"add_innate": True}

    def test_explicit_false_is_kept_for_the_tri_state_key_only(
        self, tmp_path: Path
    ) -> None:
        explicit_false = self._content(
            tmp_path / "a",
            {"can_be_generated_in_combat": False, "multiplayer_only": False},
        )
        default_null = self._content(
            tmp_path / "b", {"can_be_generated_in_combat": None}
        )

        assert explicit_false["can_be_generated_in_combat"] is False
        assert "multiplayer_only" not in explicit_false
        assert "can_be_generated_in_combat" not in default_null

    def test_nested_variant_text_is_cleaned_and_kept_last(self, tmp_path: Path) -> None:
        content = self._content(
            tmp_path,
            {
                "description": "Choose one.",
                "type_variants": [
                    {
                        "attack": {
                            "description": "Deal 12 damage.",
                            "image_url": "/static/x.webp",
                            "riders": [
                                {
                                    "id": "SAPPING",
                                    "name": "Sapping",
                                    "description": "Apply [blue]2[/blue] [gold]Weak[/gold].",
                                }
                            ],
                        }
                    }
                ],
            },
        )

        assert list(content)[-1] == "type_variants"
        assert content["type_variants"] == [
            {
                "attack": {
                    "description": "Deal 12 damage.",
                    "riders": [{"name": "Sapping", "description": "Apply 2 Weak."}],
                }
            }
        ]
