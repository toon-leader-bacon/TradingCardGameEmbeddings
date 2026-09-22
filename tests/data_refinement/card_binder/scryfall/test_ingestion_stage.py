import json
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.scryfall.ingestion_stage import (
    ScryfallCardIngestionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_ROW_WITH_ALL_ALIASES = {
    "oracle_id": "eca27964-accc-46cc-8aff-a06183e61e9c",
    "name": "Grinning Ignus",
    "multiverse_ids": [513581, 513582],
    "mtgo_id": 88685,
    "mtgo_foil_id": 88686,
    "arena_id": 76497,
    "tcgplayer_id": 235846,
    "cardmarket_id": 557248,
    "id": "cfa04897-6438-45e5-a10b-2e8afaf2b9eb",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


class TestIngestNewCards:
    def test_new_card_is_created_and_returned(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [{"oracle_id": "id-1", "name": "Lightning Bolt"}])
        binder = CardBinder()

        changed = ScryfallCardIngestionStage().ingest(raw_path, binder)

        card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert card is not None
        assert card.name == "Lightning Bolt"
        assert card.source_game == GameId.MTG
        assert changed == [card.nocab_uuid]

    def test_multiple_distinct_rows_all_created_independently(
        self, tmp_path: Path
    ) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(
            raw_path,
            [
                {"oracle_id": "id-1", "name": "Bolt"},
                {"oracle_id": "id-2", "name": "Shock"},
            ],
        )
        binder = CardBinder()

        changed = ScryfallCardIngestionStage().ingest(raw_path, binder)

        bolt = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        shock = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-2")
        assert bolt is not None and shock is not None
        assert bolt.nocab_uuid != shock.nocab_uuid
        assert set(changed) == {bolt.nocab_uuid, shock.nocab_uuid}

    def test_raises_if_row_missing_oracle_id(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [{"name": "Bolt"}])
        binder = CardBinder()

        with pytest.raises(KeyError):
            ScryfallCardIngestionStage().ingest(raw_path, binder)

    def test_raises_if_row_missing_name(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [{"oracle_id": "id-1"}])
        binder = CardBinder()

        with pytest.raises(KeyError):
            ScryfallCardIngestionStage().ingest(raw_path, binder)


class TestLeanContent:
    def test_created_card_keeps_card_fields_and_drops_noise(
        self, tmp_path: Path
    ) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        row = {
            "object": "card",
            "id": "cfa04897-6438-45e5-a10b-2e8afaf2b9eb",
            "oracle_id": "id-1",
            "name": "Lightning Bolt",
            "lang": "en",
            "released_at": "2026-09-12",
            "uri": "https://api.scryfall.com/cards/x",
            "image_uris": {"small": "https://cards.scryfall.io/small/x.jpg"},
            "mana_cost": "{R}",
            "cmc": 1.0,
            "type_line": "Instant",
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
            "colors": ["R"],
            "keywords": [],
            "prices": {"usd": "0.25", "eur": None},
            "artist": "Christopher Moeller",
            "flavor_text": "",
            "set": "m10",
            "set_name": "Magic 2010",
            "rarity": "common",
            "foil": True,
            "reserved": False,
            "digital": True,
            "all_parts": [{"name": "Other Card", "component": "combo_piece"}],
        }
        _write_jsonl(raw_path, [row])
        binder = CardBinder()

        ScryfallCardIngestionStage().ingest(raw_path, binder)

        card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert card is not None
        assert card.raw_content == {
            "name": "Lightning Bolt",
            "mana_cost": "{R}",
            "cmc": 1,
            "type_line": "Instant",
            "colors": ["R"],
            "rarity": "common",
            "set": "m10",
            "digital": True,
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
        }

    def test_key_order_puts_identity_first_and_rules_text_last(
        self, tmp_path: Path
    ) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        row = {
            "oracle_id": "id-1",
            "oracle_text": "Text.",
            "reserved": True,
            "type_line": "Instant",
            "name": "Bolt",
        }
        _write_jsonl(raw_path, [row])
        binder = CardBinder()

        ScryfallCardIngestionStage().ingest(raw_path, binder)

        card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert card is not None
        assert list(card.raw_content) == [
            "name",
            "type_line",
            "reserved",
            "oracle_text",
        ]

    def test_legalities_list_only_non_default_statuses(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        row = {
            "oracle_id": "id-1",
            "name": "Bolt",
            "legalities": {
                "standard": "not_legal",
                "modern": "legal",
                "legacy": "legal",
                "vintage": "restricted",
                "historic": "banned",
            },
        }
        _write_jsonl(raw_path, [row])
        binder = CardBinder()

        ScryfallCardIngestionStage().ingest(raw_path, binder)

        card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert card is not None
        assert card.raw_content["legalities"] == {
            "legal": ["modern", "legacy"],
            "restricted": ["vintage"],
            "banned": ["historic"],
        }

    def test_legalities_omitted_when_nothing_is_legal(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        row = {
            "oracle_id": "id-1",
            "name": "Some Token",
            "legalities": {"standard": "not_legal", "modern": "not_legal"},
        }
        _write_jsonl(raw_path, [row])
        binder = CardBinder()

        ScryfallCardIngestionStage().ingest(raw_path, binder)

        card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert card is not None
        assert "legalities" not in card.raw_content

    def test_card_faces_are_cleaned_and_ordered_per_face(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        row = {
            "oracle_id": "id-1",
            "name": "Witch Enchanter // Witch-Blessed Meadow",
            "card_faces": [
                {
                    "object": "card_face",
                    "oracle_text": "When this creature enters, destroy target artifact.",
                    "name": "Witch Enchanter",
                    "artist": "Someone",
                    "illustration_id": "7106ab4f-bd3e-4d2a-ba9c-7f223b5a0b7f",
                    "image_uris": {"small": "https://cards.scryfall.io/small/y.jpg"},
                    "mana_cost": "{3}{W}",
                    "power": "2",
                    "flavor_text": "",
                },
                {
                    "object": "card_face",
                    "name": "Witch-Blessed Meadow",
                    "type_line": "Land",
                },
            ],
        }
        _write_jsonl(raw_path, [row])
        binder = CardBinder()

        ScryfallCardIngestionStage().ingest(raw_path, binder)

        card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert card is not None
        assert card.raw_content["card_faces"] == [
            {
                "name": "Witch Enchanter",
                "mana_cost": "{3}{W}",
                "power": "2",
                "oracle_text": "When this creature enters, destroy target artifact.",
            },
            {"name": "Witch-Blessed Meadow", "type_line": "Land"},
        ]

    def test_ids_dropped_from_content_still_register_as_aliases(
        self, tmp_path: Path
    ) -> None:
        # Aliases are read from the RAW row, not from the lean content.
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [_ROW_WITH_ALL_ALIASES])
        binder = CardBinder()

        ScryfallCardIngestionStage().ingest(raw_path, binder)

        card = binder.get_by_alias(
            GameId.MTG, DataSource.SCRYFALL, _ROW_WITH_ALL_ALIASES["oracle_id"]
        )
        assert card is not None
        assert "arena_id" not in card.raw_content
        assert binder.get_by_alias(GameId.MTG, DataSource.ARENA, "76497") == card


class TestReIngestDuplicates:
    def test_unchanged_row_is_a_noop_and_preserves_uuid(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        row = {"oracle_id": "id-1", "name": "Bolt", "mana_cost": "{R}"}
        _write_jsonl(raw_path, [row])
        binder = CardBinder()
        stage = ScryfallCardIngestionStage()
        stage.ingest(raw_path, binder)
        original = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert original is not None

        second_changed = stage.ingest(raw_path, binder)

        reloaded = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert reloaded is not None
        assert reloaded.nocab_uuid == original.nocab_uuid
        assert second_changed == []

    def test_changed_row_replaces_content_in_place_keeping_uuid(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        stage = ScryfallCardIngestionStage()
        first_path = tmp_path / "first.jsonl"
        _write_jsonl(first_path, [{"oracle_id": "id-1", "name": "Bolt"}])
        stage.ingest(first_path, binder)
        original = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert original is not None

        second_path = tmp_path / "second.jsonl"
        _write_jsonl(
            second_path,
            [
                {
                    "oracle_id": "id-1",
                    "name": "Bolt",
                    "mana_cost": "{R}",
                    "type_line": "Instant",
                }
            ],
        )

        changed = stage.ingest(second_path, binder)

        updated = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert updated is not None
        assert updated.nocab_uuid == original.nocab_uuid
        assert updated.raw_content == {
            "name": "Bolt",
            "mana_cost": "{R}",
            "type_line": "Instant",
        }
        assert changed == [original.nocab_uuid]

    def test_shorter_incoming_row_still_wins(self, tmp_path: Path) -> None:
        # The dump is authoritative: errata that shortens rules text must
        # not lose to a stale, longer stored row.
        binder = CardBinder()
        stage = ScryfallCardIngestionStage()
        old_path = tmp_path / "old.jsonl"
        _write_jsonl(
            old_path,
            [{"oracle_id": "id-1", "name": "Bolt", "oracle_text": "Long old text."}],
        )
        stage.ingest(old_path, binder)
        original = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert original is not None

        new_path = tmp_path / "new.jsonl"
        _write_jsonl(
            new_path, [{"oracle_id": "id-1", "name": "Bolt", "oracle_text": "New."}]
        )

        changed = stage.ingest(new_path, binder)

        updated = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert updated is not None
        assert updated.raw_content["oracle_text"] == "New."
        assert updated.nocab_uuid == original.nocab_uuid
        assert changed == [original.nocab_uuid]

    def test_row_differing_only_in_noise_is_a_noop(self, tmp_path: Path) -> None:
        # Prices, image URLs and ids change between dumps; the card does not.
        binder = CardBinder()
        stage = ScryfallCardIngestionStage()
        first_path = tmp_path / "first.jsonl"
        _write_jsonl(
            first_path,
            [{"oracle_id": "id-1", "name": "Bolt", "prices": {"usd": "0.25"}}],
        )
        stage.ingest(first_path, binder)
        original = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert original is not None

        second_path = tmp_path / "second.jsonl"
        _write_jsonl(
            second_path,
            [
                {
                    "oracle_id": "id-1",
                    "name": "Bolt",
                    "prices": {"usd": "0.31"},
                    "image_uris": {"small": "https://x.io/a.jpg"},
                }
            ],
        )

        changed = stage.ingest(second_path, binder)

        assert changed == []
        assert binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1") == original


class TestAliasRegistration:
    def test_all_secondary_aliases_resolve_after_create(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [_ROW_WITH_ALL_ALIASES])
        binder = CardBinder()

        ScryfallCardIngestionStage().ingest(raw_path, binder)

        card = binder.get_by_alias(
            GameId.MTG, DataSource.SCRYFALL, _ROW_WITH_ALL_ALIASES["oracle_id"]
        )
        assert card is not None
        assert binder.get_by_alias(GameId.MTG, DataSource.ARENA, "76497") == card
        assert binder.get_by_alias(GameId.MTG, DataSource.MTGO, "88685") == card
        assert binder.get_by_alias(GameId.MTG, DataSource.MTGO, "88686") == card
        assert binder.get_by_alias(GameId.MTG, DataSource.GATHERER, "513581") == card
        assert binder.get_by_alias(GameId.MTG, DataSource.GATHERER, "513582") == card

    def test_aliases_still_registered_on_noop_duplicate_branch(
        self, tmp_path: Path
    ) -> None:
        # Aliases must be re-registered even when content doesn't
        # change — a losing/no-op row's identifier must never become
        # a dead end (plans/card_binder_v2.md's "Open risks"). arena_id is
        # dropped from raw_content as noise, so the two rows have identical
        # content and the second takes the no-op branch.
        binder = CardBinder()
        stage = ScryfallCardIngestionStage()
        first_path = tmp_path / "first.jsonl"
        _write_jsonl(first_path, [{"oracle_id": "id-1", "name": "Bolt"}])
        stage.ingest(first_path, binder)

        second_path = tmp_path / "second.jsonl"
        _write_jsonl(
            second_path,
            [{"oracle_id": "id-1", "name": "Bolt", "arena_id": 76497}],
        )
        changed = stage.ingest(second_path, binder)

        card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert changed == []
        assert binder.get_by_alias(GameId.MTG, DataSource.ARENA, "76497") == card

    def test_missing_arena_id_is_tolerated(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        row = {"oracle_id": "id-1", "name": "Candles of Leng", "mtgo_id": 25639}
        _write_jsonl(raw_path, [row])
        binder = CardBinder()

        ScryfallCardIngestionStage().ingest(raw_path, binder)

        card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert binder.get_by_alias(GameId.MTG, DataSource.MTGO, "25639") == card
        assert binder.get_by_alias(GameId.MTG, DataSource.ARENA, "76497") is None


class TestExtractAliases:
    def test_extracts_every_present_alias_field(self) -> None:
        aliases = set(
            ScryfallCardIngestionStage()._extract_aliases(_ROW_WITH_ALL_ALIASES)
        )

        assert aliases == {
            (DataSource.ARENA, "76497"),
            (DataSource.MTGO, "88685"),
            (DataSource.MTGO, "88686"),
            (DataSource.GATHERER, "513581"),
            (DataSource.GATHERER, "513582"),
        }

    def test_row_with_no_alias_fields_returns_empty_list(self) -> None:
        aliases = ScryfallCardIngestionStage()._extract_aliases(
            {"oracle_id": "id-1", "name": "Bolt"}
        )

        assert aliases == []
