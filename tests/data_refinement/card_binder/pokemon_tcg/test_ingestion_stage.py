import json
from pathlib import Path

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
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
    def test_creates_one_card_per_row_in_one_file(self, tmp_path: Path) -> None:
        _write_set_file(
            tmp_path / "base4.json",
            [
                {"id": "base4-1", "name": "Alakazam"},
                {"id": "base4-2", "name": "Chansey"},
            ],
        )
        binder = CardBinder()

        changed = PokemonTcgCardIngestionStage().ingest(tmp_path, binder)

        assert len(changed) == 2
        assert set(binder.all_uuids(GameId.POKEMON)) == set(changed)

    def test_creates_across_multiple_set_files(self, tmp_path: Path) -> None:
        _write_set_file(
            tmp_path / "base4.json", [{"id": "base4-1", "name": "Alakazam"}]
        )
        _write_set_file(tmp_path / "bw1.json", [{"id": "bw1-1", "name": "Emboar"}])
        binder = CardBinder()

        PokemonTcgCardIngestionStage().ingest(tmp_path, binder)

        assert binder.get_by_name(GameId.POKEMON, "Alakazam") != []
        assert binder.get_by_name(GameId.POKEMON, "Emboar") != []

    def test_maps_id_to_provenance_source_id(self, tmp_path: Path) -> None:
        _write_set_file(tmp_path / "base4.json", [_REAL_ROW])
        binder = CardBinder()

        PokemonTcgCardIngestionStage().ingest(tmp_path, binder)

        card = binder.get_by_alias(GameId.POKEMON, DataSource.POKEMON_TCG, "base4-1")
        assert card is not None
        assert card.provenance.source_id == _REAL_ROW["id"]
        assert card.provenance.data_source == DataSource.POKEMON_TCG

    def test_raw_content_is_the_entire_row_unmodified(self, tmp_path: Path) -> None:
        _write_set_file(tmp_path / "base4.json", [_REAL_ROW])
        binder = CardBinder()

        PokemonTcgCardIngestionStage().ingest(tmp_path, binder)

        card = binder.get_by_alias(GameId.POKEMON, DataSource.POKEMON_TCG, "base4-1")
        assert card.raw_content == _REAL_ROW

    def test_each_card_gets_a_distinct_uuid(self, tmp_path: Path) -> None:
        _write_set_file(
            tmp_path / "base4.json",
            [
                {"id": "base4-1", "name": "Alakazam"},
                {"id": "base4-2", "name": "Chansey"},
            ],
        )
        binder = CardBinder()

        PokemonTcgCardIngestionStage().ingest(tmp_path, binder)

        alakazam = binder.get_by_alias(
            GameId.POKEMON, DataSource.POKEMON_TCG, "base4-1"
        )
        chansey = binder.get_by_alias(GameId.POKEMON, DataSource.POKEMON_TCG, "base4-2")
        assert alakazam.nocab_uuid != chansey.nocab_uuid

    def test_same_name_different_set_stays_distinct(self, tmp_path: Path) -> None:
        # Same Pokemon name reprinted in a genuinely different set —
        # these are distinct cards under this project's own identity
        # rule (name, set code), confirmed against real
        # cross-era Pokemon reprints.
        _write_set_file(
            tmp_path / "base4.json", [{"id": "base4-1", "name": "Alakazam"}]
        )
        _write_set_file(tmp_path / "neo1.json", [{"id": "neo1-1", "name": "Alakazam"}])
        binder = CardBinder()

        changed = PokemonTcgCardIngestionStage().ingest(tmp_path, binder)

        assert len(changed) == 2
        matches = binder.get_by_name(GameId.POKEMON, "Alakazam")
        assert len(matches) == 2
        assert {c.nocab_uuid for c in matches} == set(changed)

    def test_same_name_same_set_collapses_to_one_card(self, tmp_path: Path) -> None:
        # Alternate-art/rarity reprints within one set share (name,
        # set code) and identical rules text — confirmed against real
        # data (e.g. "Mew V" appears 3x in swsh8.json with identical
        # attacks). These collapse to one card, same relationship as
        # Scryfall's multiple printings of one oracle_id.
        _write_set_file(
            tmp_path / "swsh8.json",
            [
                {"id": "swsh8-113", "name": "Mew V", "hp": "210"},
                {"id": "swsh8-250", "name": "Mew V", "hp": "210"},
            ],
        )
        binder = CardBinder()

        PokemonTcgCardIngestionStage().ingest(tmp_path, binder)

        matches = binder.get_by_name(GameId.POKEMON, "Mew V")
        assert len(matches) == 1

    def test_raises_if_raw_path_does_not_exist(self, tmp_path: Path) -> None:
        binder = CardBinder()
        with pytest.raises(ValueError):
            PokemonTcgCardIngestionStage().ingest(tmp_path / "missing", binder)

    def test_raises_if_raw_path_is_not_a_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir.json"
        file_path.write_text("[]")
        binder = CardBinder()

        with pytest.raises(ValueError):
            PokemonTcgCardIngestionStage().ingest(file_path, binder)

    def test_raises_if_no_json_files_found(self, tmp_path: Path) -> None:
        (tmp_path / "decks").mkdir()
        binder = CardBinder()

        with pytest.raises(ValueError):
            PokemonTcgCardIngestionStage().ingest(tmp_path, binder)


class TestReIngestDuplicates:
    def test_unchanged_row_is_a_noop_and_preserves_uuid(self, tmp_path: Path) -> None:
        binder = CardBinder()
        stage = PokemonTcgCardIngestionStage()
        _write_set_file(tmp_path / "base4.json", [_REAL_ROW])
        stage.ingest(tmp_path, binder)
        original = binder.get_by_alias(
            GameId.POKEMON, DataSource.POKEMON_TCG, "base4-1"
        )

        second_dir = tmp_path / "second"
        second_dir.mkdir()
        _write_set_file(second_dir / "base4.json", [_REAL_ROW])
        changed = stage.ingest(second_dir, binder)

        reloaded = binder.get_by_alias(
            GameId.POKEMON, DataSource.POKEMON_TCG, "base4-1"
        )
        assert reloaded.nocab_uuid == original.nocab_uuid
        assert changed == []

    def test_richer_alternate_art_updates_content_in_place(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        stage = PokemonTcgCardIngestionStage()
        sparse_dir = tmp_path / "sparse"
        sparse_dir.mkdir()
        _write_set_file(
            sparse_dir / "swsh8.json", [{"id": "swsh8-113", "name": "Mew V"}]
        )
        stage.ingest(sparse_dir, binder)
        original = binder.get_by_alias(
            GameId.POKEMON, DataSource.POKEMON_TCG, "swsh8-113"
        )

        richer_dir = tmp_path / "richer"
        richer_dir.mkdir()
        richer_row = {
            "id": "swsh8-250",
            "name": "Mew V",
            "hp": "210",
            "attacks": [{"name": "Energy Mix"}],
        }
        _write_set_file(richer_dir / "swsh8.json", [richer_row])

        changed = stage.ingest(richer_dir, binder)

        updated = binder.get_by_alias(
            GameId.POKEMON, DataSource.POKEMON_TCG, "swsh8-113"
        )
        assert updated.nocab_uuid == original.nocab_uuid
        assert updated.raw_content == richer_row
        assert changed == [original.nocab_uuid]
        # Both printing ids resolve to the same, now-merged card.
        assert (
            binder.get_by_alias(
                GameId.POKEMON, DataSource.POKEMON_TCG, "swsh8-250"
            ).nocab_uuid
            == original.nocab_uuid
        )
