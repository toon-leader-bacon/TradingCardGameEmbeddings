import json
from pathlib import Path

import pytest

from src.data_refinement.card_binder.pokemon_tcg.ingestion_stage import (
    PokemonTcgCardIngestionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_REAL_ROW = {
    "id": "base4-1",
    "name": "Alakazam",
    "supertype": "Pokémon",
    "subtypes": ["Stage 2"],
    "hp": "80",
    "types": ["Psychic"],
    "nationalPokedexNumbers": [65],
}


def _write_set_file(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows))


class TestIngest:
    def test_parses_one_candidate_per_card_in_one_file(self, tmp_path: Path) -> None:
        _write_set_file(
            tmp_path / "base4.json",
            [
                {"id": "base4-1", "name": "Alakazam"},
                {"id": "base4-2", "name": "Chansey"},
            ],
        )

        candidates = PokemonTcgCardIngestionStage().ingest(tmp_path, GameId.POKEMON)

        assert len(candidates) == 2
        assert {c.card.name for c in candidates} == {"Alakazam", "Chansey"}

    def test_parses_across_multiple_set_files(self, tmp_path: Path) -> None:
        _write_set_file(tmp_path / "base4.json", [{"id": "base4-1", "name": "Alakazam"}])
        _write_set_file(tmp_path / "bw1.json", [{"id": "bw1-1", "name": "Emboar"}])

        candidates = PokemonTcgCardIngestionStage().ingest(tmp_path, GameId.POKEMON)

        assert {c.card.name for c in candidates} == {"Alakazam", "Emboar"}

    def test_maps_id_to_provenance_source_id(self, tmp_path: Path) -> None:
        _write_set_file(tmp_path / "base4.json", [_REAL_ROW])

        candidates = PokemonTcgCardIngestionStage().ingest(tmp_path, GameId.POKEMON)

        assert candidates[0].card.provenance.source_id == _REAL_ROW["id"]
        assert candidates[0].card.provenance.data_source == DataSource.POKEMON_TCG

    def test_raw_content_is_the_entire_row_unmodified(self, tmp_path: Path) -> None:
        _write_set_file(tmp_path / "base4.json", [_REAL_ROW])

        candidates = PokemonTcgCardIngestionStage().ingest(tmp_path, GameId.POKEMON)

        assert candidates[0].card.raw_content == _REAL_ROW

    def test_threads_source_game_through_unchanged(self, tmp_path: Path) -> None:
        _write_set_file(tmp_path / "base4.json", [{"id": "base4-1", "name": "Alakazam"}])

        candidates = PokemonTcgCardIngestionStage().ingest(tmp_path, GameId.POKEMON)

        assert candidates[0].card.source_game == GameId.POKEMON

    def test_each_card_gets_a_distinct_uuid(self, tmp_path: Path) -> None:
        _write_set_file(
            tmp_path / "base4.json",
            [
                {"id": "base4-1", "name": "Alakazam"},
                {"id": "base4-2", "name": "Chansey"},
            ],
        )

        candidates = PokemonTcgCardIngestionStage().ingest(tmp_path, GameId.POKEMON)

        assert candidates[0].card.nocab_uuid != candidates[1].card.nocab_uuid

    def test_does_not_filter_or_deduplicate_by_name(self, tmp_path: Path) -> None:
        # Same Pokémon reprinted across sets — both should become
        # separate IngestedCandidates; collapsing them is CardBinder's
        # job, not this stage's.
        _write_set_file(tmp_path / "base4.json", [{"id": "base4-1", "name": "Alakazam"}])
        _write_set_file(tmp_path / "neo1.json", [{"id": "neo1-1", "name": "Alakazam"}])

        candidates = PokemonTcgCardIngestionStage().ingest(tmp_path, GameId.POKEMON)

        assert len(candidates) == 2
        assert candidates[0].card.nocab_uuid != candidates[1].card.nocab_uuid

    def test_aliases_are_always_empty(self, tmp_path: Path) -> None:
        _write_set_file(tmp_path / "base4.json", [_REAL_ROW])

        candidates = PokemonTcgCardIngestionStage().ingest(tmp_path, GameId.POKEMON)

        assert candidates[0].aliases == []

    def test_raises_if_raw_path_does_not_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            PokemonTcgCardIngestionStage().ingest(tmp_path / "missing", GameId.POKEMON)

    def test_raises_if_raw_path_is_not_a_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir.json"
        file_path.write_text("[]")

        with pytest.raises(ValueError):
            PokemonTcgCardIngestionStage().ingest(file_path, GameId.POKEMON)

    def test_raises_if_no_json_files_found(self, tmp_path: Path) -> None:
        (tmp_path / "decks").mkdir()

        with pytest.raises(ValueError):
            PokemonTcgCardIngestionStage().ingest(tmp_path, GameId.POKEMON)
