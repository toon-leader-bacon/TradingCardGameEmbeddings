from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.alias_ledger import AliasLedger
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class TestResolve:
    def test_found(self) -> None:
        ledger = AliasLedger()
        nocab_uuid = uuid4()
        ledger.register(GameId.MTG, DataSource.ARENA, "76497", nocab_uuid)

        assert ledger.resolve(GameId.MTG, DataSource.ARENA, "76497") == nocab_uuid

    def test_not_found(self) -> None:
        ledger = AliasLedger()

        assert ledger.resolve(GameId.MTG, DataSource.ARENA, "nonexistent") is None

    def test_same_source_id_different_data_source_does_not_collide(self) -> None:
        ledger = AliasLedger()
        arena_uuid = uuid4()
        mtgo_uuid = uuid4()
        ledger.register(GameId.MTG, DataSource.ARENA, "12345", arena_uuid)
        ledger.register(GameId.MTG, DataSource.MTGO, "12345", mtgo_uuid)

        assert ledger.resolve(GameId.MTG, DataSource.ARENA, "12345") == arena_uuid
        assert ledger.resolve(GameId.MTG, DataSource.MTGO, "12345") == mtgo_uuid

    def test_same_source_id_different_game_does_not_collide(self) -> None:
        ledger = AliasLedger()
        mtg_uuid = uuid4()
        pokemon_uuid = uuid4()
        ledger.register(GameId.MTG, DataSource.SCRYFALL, "shared-id", mtg_uuid)
        ledger.register(GameId.POKEMON, DataSource.SCRYFALL, "shared-id", pokemon_uuid)

        assert ledger.resolve(GameId.MTG, DataSource.SCRYFALL, "shared-id") == mtg_uuid
        assert (
            ledger.resolve(GameId.POKEMON, DataSource.SCRYFALL, "shared-id")
            == pokemon_uuid
        )


class TestRegister:
    def test_overwrites_prior_mapping_for_the_same_key(self) -> None:
        ledger = AliasLedger()
        first_uuid = uuid4()
        second_uuid = uuid4()
        ledger.register(GameId.MTG, DataSource.ARENA, "76497", first_uuid)

        ledger.register(GameId.MTG, DataSource.ARENA, "76497", second_uuid)

        assert ledger.resolve(GameId.MTG, DataSource.ARENA, "76497") == second_uuid


class TestSaveAndLoad:
    def test_round_trips(self, tmp_path: Path) -> None:
        ledger = AliasLedger()
        nocab_uuid = uuid4()
        ledger.register(GameId.MTG, DataSource.ARENA, "76497", nocab_uuid)
        path = tmp_path / "mtg.alias_ledger.jsonl"

        ledger.save(path, GameId.MTG)
        loaded = AliasLedger.load(path, GameId.MTG)

        assert loaded.resolve(GameId.MTG, DataSource.ARENA, "76497") == nocab_uuid

    def test_save_writes_only_requested_games_subset(self, tmp_path: Path) -> None:
        ledger = AliasLedger()
        ledger.register(GameId.MTG, DataSource.ARENA, "76497", uuid4())
        ledger.register(GameId.POKEMON, DataSource.SCRYFALL, "poke-1", uuid4())
        path = tmp_path / "mtg.alias_ledger.jsonl"

        ledger.save(path, GameId.MTG)
        loaded = AliasLedger.load(path, GameId.MTG)

        assert loaded.resolve(GameId.MTG, DataSource.ARENA, "76497") is not None
        assert loaded.resolve(GameId.POKEMON, DataSource.SCRYFALL, "poke-1") is None

    def test_save_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        ledger = AliasLedger()
        ledger.register(GameId.MTG, DataSource.ARENA, "76497", uuid4())
        nested_path = tmp_path / "does" / "not" / "exist" / "mtg.alias_ledger.jsonl"

        ledger.save(nested_path, GameId.MTG)

        assert nested_path.exists()

    def test_load_of_empty_file_returns_usable_empty_ledger(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.alias_ledger.jsonl"
        path.write_text("")

        loaded = AliasLedger.load(path, GameId.MTG)

        assert loaded.resolve(GameId.MTG, DataSource.ARENA, "76497") is None
