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

    def test_multiple_distinct_rows_all_created_independently(self, tmp_path: Path) -> None:
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

    def test_richer_row_updates_content_in_place(self, tmp_path: Path) -> None:
        binder = CardBinder()
        stage = ScryfallCardIngestionStage()
        sparse_path = tmp_path / "sparse.jsonl"
        _write_jsonl(sparse_path, [{"oracle_id": "id-1", "name": "Bolt"}])
        stage.ingest(sparse_path, binder)
        original = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert original is not None

        richer_path = tmp_path / "richer.jsonl"
        richer_row = {
            "oracle_id": "id-1",
            "name": "Bolt",
            "mana_cost": "{R}",
            "type_line": "Instant",
        }
        _write_jsonl(richer_path, [richer_row])

        changed = stage.ingest(richer_path, binder)

        updated = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert updated is not None
        assert updated.nocab_uuid == original.nocab_uuid
        assert updated.raw_content == richer_row
        assert changed == [original.nocab_uuid]

    def test_less_rich_row_leaves_existing_content_untouched(self, tmp_path: Path) -> None:
        binder = CardBinder()
        stage = ScryfallCardIngestionStage()
        richer_path = tmp_path / "richer.jsonl"
        richer_row = {
            "oracle_id": "id-1",
            "name": "Bolt",
            "mana_cost": "{R}",
            "type_line": "Instant",
        }
        _write_jsonl(richer_path, [richer_row])
        stage.ingest(richer_path, binder)
        original = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert original is not None

        sparse_path = tmp_path / "sparse.jsonl"
        _write_jsonl(sparse_path, [{"oracle_id": "id-1", "name": "Bolt"}])

        changed = stage.ingest(sparse_path, binder)

        unchanged = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
        assert unchanged is not None
        assert unchanged.nocab_uuid == original.nocab_uuid
        assert unchanged.raw_content == richer_row
        assert changed == []


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

    def test_aliases_still_registered_on_noop_duplicate_branch(self, tmp_path: Path) -> None:
        # Aliases must be re-registered even when content doesn't
        # change — a losing/no-op row's identifier must never become
        # a dead end (plans/card_binder_v2.md's "Open risks").
        binder = CardBinder()
        stage = ScryfallCardIngestionStage()
        first_path = tmp_path / "first.jsonl"
        _write_jsonl(first_path, [{"oracle_id": "id-1", "name": "Bolt", "a": "x" * 50}])
        stage.ingest(first_path, binder)

        second_path = tmp_path / "second.jsonl"
        _write_jsonl(
            second_path,
            [{"oracle_id": "id-1", "name": "Bolt", "arena_id": 76497}],
        )
        stage.ingest(second_path, binder)

        card = binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "id-1")
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
        aliases = set(ScryfallCardIngestionStage()._extract_aliases(_ROW_WITH_ALL_ALIASES))

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
