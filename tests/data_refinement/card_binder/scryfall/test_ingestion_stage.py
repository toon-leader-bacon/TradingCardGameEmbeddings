import json
from pathlib import Path

from src.data_refinement.card_binder.scryfall.ingestion_stage import (
    ScryfallCardIngestionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_REAL_ROW = {
    "object": "card",
    "id": "a471b306-4941-4e46-a0cb-d92895c16f8a",
    "oracle_id": "00037840-6089-42ec-8c5c-281f9f474504",
    "name": "Nissa, Worldsoul Speaker",
    "mana_cost": "{3}{G}",
    "type_line": "Legendary Creature — Elf Druid",
    "colors": ["G"],
}

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
    "set_id": "541c3c28-8747-40e5-a231-8e8f33234859",
    "card_back_id": "0aeebaf5-8c7d-4636-9e82-8c27447861f7",
    "illustration_id": "94c5ee02-12d3-47e9-9955-e15a80309ba5",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


class TestIngest:
    def test_parses_one_candidate_per_line(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(
            raw_path,
            [
                {"oracle_id": "id-1", "name": "Lightning Bolt"},
                {"oracle_id": "id-2", "name": "Counterspell"},
            ],
        )

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        assert len(candidates) == 2
        assert [c.card.name for c in candidates] == ["Lightning Bolt", "Counterspell"]
        assert [c.card.provenance.source_id for c in candidates] == ["id-1", "id-2"]

    def test_maps_oracle_id_to_provenance_not_scryfall_id(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [_REAL_ROW])

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        assert candidates[0].card.provenance.source_id == _REAL_ROW["oracle_id"]
        assert candidates[0].card.provenance.source_id != _REAL_ROW["id"]
        assert candidates[0].card.provenance.data_source == DataSource.SCRYFALL

    def test_raw_content_is_the_entire_row_unmodified(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [_REAL_ROW])

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        assert candidates[0].card.raw_content == _REAL_ROW

    def test_threads_source_game_through_unchanged(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [{"oracle_id": "id-1", "name": "Bolt"}])

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        assert candidates[0].card.source_game == GameId.MTG

    def test_each_card_gets_a_distinct_uuid(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(
            raw_path,
            [
                {"oracle_id": "id-1", "name": "Bolt"},
                {"oracle_id": "id-2", "name": "Shock"},
            ],
        )

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        assert candidates[0].card.nocab_uuid != candidates[1].card.nocab_uuid

    def test_does_not_filter_or_deduplicate_by_name(self, tmp_path: Path) -> None:
        # Two distinct token printings sharing a name — both should
        # become separate IngestedCandidates; dedup is CardBinder's
        # job, not this stage's.
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(
            raw_path,
            [
                {"oracle_id": "token-1", "name": "Storm Crow", "layout": "token"},
                {"oracle_id": "token-2", "name": "Storm Crow", "layout": "token"},
            ],
        )

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        assert len(candidates) == 2
        assert candidates[0].card.nocab_uuid != candidates[1].card.nocab_uuid

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "empty.jsonl"
        raw_path.write_text("")

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        assert candidates == []


class TestExtractAliases:
    def test_extracts_every_present_alias_field(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [_ROW_WITH_ALL_ALIASES])

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        aliases = {(a.data_source, a.source_id) for a in candidates[0].aliases}
        assert aliases == {
            (DataSource.ARENA, "76497"),
            (DataSource.MTGO, "88685"),
            (DataSource.MTGO, "88686"),
            (DataSource.GATHERER, "513581"),
            (DataSource.GATHERER, "513582"),
        }

    def test_missing_arena_id_is_tolerated(self, tmp_path: Path) -> None:
        # Confirmed real case: Arena-illegal cards have no arena_id field.
        raw_path = tmp_path / "oracle-cards.jsonl"
        row = {"oracle_id": "id-1", "name": "Candles of Leng", "mtgo_id": 25639}
        _write_jsonl(raw_path, [row])

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        aliases = {(a.data_source, a.source_id) for a in candidates[0].aliases}
        assert aliases == {(DataSource.MTGO, "25639")}

    def test_row_with_no_alias_fields_returns_empty_aliases(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [{"oracle_id": "id-1", "name": "Bolt"}])

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        assert candidates[0].aliases == []

    def test_excluded_fields_never_become_aliases(self, tmp_path: Path) -> None:
        # id/set_id/card_back_id/illustration_id/tcgplayer_id/cardmarket_id
        # are deliberately not identity-bearing — see this stage's
        # module docstring.
        raw_path = tmp_path / "oracle-cards.jsonl"
        _write_jsonl(raw_path, [_ROW_WITH_ALL_ALIASES])

        candidates = ScryfallCardIngestionStage().ingest(raw_path, GameId.MTG)

        excluded_values = {
            _ROW_WITH_ALL_ALIASES["id"],
            _ROW_WITH_ALL_ALIASES["set_id"],
            _ROW_WITH_ALL_ALIASES["card_back_id"],
            _ROW_WITH_ALL_ALIASES["illustration_id"],
            str(_ROW_WITH_ALL_ALIASES["tcgplayer_id"]),
            str(_ROW_WITH_ALL_ALIASES["cardmarket_id"]),
        }
        alias_ids = {a.source_id for a in candidates[0].aliases}
        assert alias_ids.isdisjoint(excluded_values)
