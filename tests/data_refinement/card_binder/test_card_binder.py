from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(
    name: str,
    source_id: str,
    raw_content: dict,
    source_game: GameId = GameId.MTG,
    data_source: DataSource = DataSource.SCRYFALL,
) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=source_game,
        name=name,
        raw_content=raw_content,
        provenance=Provenance(
            data_source=data_source,
            source_id=source_id,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestCreate:
    def test_inserts_new_card(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})

        result = binder.create(card)

        assert result == card
        assert binder.get_by_uuid(card.nocab_uuid) == card

    def test_raises_on_duplicate_uuid(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        with pytest.raises(ValueError):
            binder.create(card)


class TestUpdate:
    def test_shallow_merges_raw_content_patch(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1, "b": 2})
        binder.create(card)

        updated = binder.update(card.nocab_uuid, raw_content_patch={"b": 20, "c": 3})

        assert updated.raw_content == {"a": 1, "b": 20, "c": 3}
        assert updated.nocab_uuid == card.nocab_uuid

    def test_overrides_name_when_given(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        updated = binder.update(card.nocab_uuid, name="Lightning Bolt")

        assert updated.name == "Lightning Bolt"
        assert binder.get_by_name(GameId.MTG, "Bolt") == []
        assert binder.get_by_name(GameId.MTG, "Lightning Bolt") == [updated]

    def test_overrides_provenance_when_given(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)
        new_provenance = Provenance(
            data_source=DataSource.ARENA,
            source_id="76497",
            fetched_at=datetime.now(timezone.utc),
        )

        updated = binder.update(card.nocab_uuid, provenance=new_provenance)

        assert updated.provenance == new_provenance

    def test_leaves_fields_untouched_when_not_given(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        updated = binder.update(card.nocab_uuid)

        assert updated.raw_content == card.raw_content
        assert updated.name == card.name
        assert updated.provenance == card.provenance

    def test_raises_if_uuid_not_stored(self) -> None:
        binder = CardBinder()

        with pytest.raises(KeyError):
            binder.update(uuid4(), raw_content_patch={"a": 1})


class TestReplace:
    def test_fully_swaps_content(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)
        replacement = GenericCard(
            nocab_uuid=card.nocab_uuid,
            source_game=GameId.MTG,
            name="Bolt",
            raw_content={"z": 9},
            provenance=Provenance(
                data_source=DataSource.ARENA,
                source_id="76497",
                fetched_at=datetime.now(timezone.utc),
            ),
        )

        result = binder.replace(card.nocab_uuid, replacement)

        assert result == replacement
        assert binder.get_by_uuid(card.nocab_uuid) == replacement

    def test_raises_on_uuid_mismatch(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)
        mismatched = _card("Bolt", "src-2", {"a": 1})

        with pytest.raises(ValueError):
            binder.replace(card.nocab_uuid, mismatched)

    def test_raises_if_uuid_not_stored(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})

        with pytest.raises(KeyError):
            binder.replace(card.nocab_uuid, card)

    def test_source_game_change_updates_name_index(self) -> None:
        binder = CardBinder()
        card = _card("Boost", "src-1", {"a": 1}, source_game=GameId.MTG)
        binder.create(card)

        moved = replace(card, source_game=GameId.POKEMON)
        binder.replace(card.nocab_uuid, moved)

        assert binder.get_by_name(GameId.MTG, "Boost") == []
        assert binder.get_by_name(GameId.POKEMON, "Boost") == [moved]


class TestDelete:
    def test_removes_card(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        binder.delete(card.nocab_uuid)

        assert binder.get_by_uuid(card.nocab_uuid) is None

    def test_removes_card_from_name_index(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        binder.delete(card.nocab_uuid)

        assert binder.get_by_name(GameId.MTG, "Bolt") == []

    def test_raises_if_uuid_not_stored(self) -> None:
        binder = CardBinder()

        with pytest.raises(KeyError):
            binder.delete(uuid4())


class TestGetByUuid:
    def test_found(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        assert binder.get_by_uuid(card.nocab_uuid) == card

    def test_not_found(self) -> None:
        binder = CardBinder()

        assert binder.get_by_uuid(uuid4()) is None


class TestGetByName:
    def test_found(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        assert binder.get_by_name(GameId.MTG, "Bolt") == [card]

    def test_not_found_returns_empty_list(self) -> None:
        binder = CardBinder()

        assert binder.get_by_name(GameId.MTG, "Nonexistent") == []

    def test_multiple_cards_can_share_a_name(self) -> None:
        binder = CardBinder()
        strike_ironclad = _card("Strike", "STRIKE_IRONCLAD", {"color": "ironclad"})
        strike_silent = _card("Strike", "STRIKE_SILENT", {"color": "silent"})
        binder.create(strike_ironclad)
        binder.create(strike_silent)

        matches = binder.get_by_name(GameId.MTG, "Strike")

        assert {c.nocab_uuid for c in matches} == {
            strike_ironclad.nocab_uuid,
            strike_silent.nocab_uuid,
        }

    def test_same_name_different_game_does_not_collide(self) -> None:
        binder = CardBinder()
        mtg_card = _card("Boost", "src-1", {"a": 1}, source_game=GameId.MTG)
        pokemon_card = _card("Boost", "src-2", {"a": 1}, source_game=GameId.POKEMON)
        binder.create(mtg_card)
        binder.create(pokemon_card)

        assert binder.get_by_name(GameId.MTG, "Boost") == [mtg_card]
        assert binder.get_by_name(GameId.POKEMON, "Boost") == [pokemon_card]


class TestGetByNameSingle:
    def test_returns_the_one_match(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        assert binder.get_by_name_single(GameId.MTG, "Bolt") == card

    def test_returns_none_if_no_match(self) -> None:
        binder = CardBinder()

        assert binder.get_by_name_single(GameId.MTG, "Nonexistent") is None

    def test_strict_raises_on_ambiguity(self) -> None:
        binder = CardBinder()
        binder.create(_card("Strike", "STRIKE_IRONCLAD", {}))
        binder.create(_card("Strike", "STRIKE_SILENT", {}))

        with pytest.raises(ValueError):
            binder.get_by_name_single(GameId.MTG, "Strike", strict=True)

    def test_non_strict_returns_one_of_the_matches(self) -> None:
        binder = CardBinder()
        ironclad = _card("Strike", "STRIKE_IRONCLAD", {})
        silent = _card("Strike", "STRIKE_SILENT", {})
        binder.create(ironclad)
        binder.create(silent)

        result = binder.get_by_name_single(GameId.MTG, "Strike", strict=False)

        assert result is not None
        assert result.nocab_uuid in {ironclad.nocab_uuid, silent.nocab_uuid}


class TestGetByNameRegex:
    def test_exact_name_as_pattern_matches(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        assert binder.get_by_name_regex(GameId.MTG, "^Bolt$") == [card]

    def test_mdfc_fallback_pattern_matches_combined_name(self) -> None:
        binder = CardBinder()
        card = _card("Bruce Banner // The Incredible Hulk", "src-1", {"a": 1})
        binder.create(card)

        matches = binder.get_by_name_regex(GameId.MTG, "^Bruce Banner( //.*)?$")

        assert matches == [card]

    def test_no_match_returns_empty_list(self) -> None:
        binder = CardBinder()
        binder.create(_card("Bolt", "src-1", {"a": 1}))

        assert binder.get_by_name_regex(GameId.MTG, "^Shock$") == []

    def test_multiple_matches_all_returned(self) -> None:
        binder = CardBinder()
        bolt = _card("Bolt", "src-1", {"a": 1})
        bolts = _card("Bolts", "src-2", {"a": 1})
        binder.create(bolt)
        binder.create(bolts)

        matches = binder.get_by_name_regex(GameId.MTG, "^Bolt")

        assert {c.nocab_uuid for c in matches} == {bolt.nocab_uuid, bolts.nocab_uuid}

    def test_respects_source_game_filtering(self) -> None:
        binder = CardBinder()
        binder.create(_card("Boost", "src-1", {"a": 1}, source_game=GameId.MTG))

        assert binder.get_by_name_regex(GameId.POKEMON, "^Boost$") == []


class TestGetByAlias:
    def test_found(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)
        binder.register_alias(GameId.MTG, DataSource.SCRYFALL, "src-1", card.nocab_uuid)

        assert binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-1") == card

    def test_not_found(self) -> None:
        binder = CardBinder()

        assert (
            binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "nonexistent") is None
        )


class TestRegisterAlias:
    def test_raises_for_unknown_nocab_uuid(self) -> None:
        binder = CardBinder()

        with pytest.raises(ValueError):
            binder.register_alias(GameId.MTG, DataSource.ARENA, "76497", uuid4())

    def test_extra_alias_resolves_to_card(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)

        binder.register_alias(GameId.MTG, DataSource.ARENA, "76497", card.nocab_uuid)

        assert binder.get_by_alias(GameId.MTG, DataSource.ARENA, "76497") == card


class TestAllUuids:
    def test_no_filter_returns_every_uuid(self) -> None:
        binder = CardBinder()
        mtg_card = _card("Bolt", "src-1", {"a": 1}, source_game=GameId.MTG)
        pokemon_card = _card(
            "Charmander", "src-2", {"a": 1}, source_game=GameId.POKEMON
        )
        binder.create(mtg_card)
        binder.create(pokemon_card)

        assert set(binder.all_uuids()) == {mtg_card.nocab_uuid, pokemon_card.nocab_uuid}

    def test_filter_by_game(self) -> None:
        binder = CardBinder()
        mtg_card = _card("Bolt", "src-1", {"a": 1}, source_game=GameId.MTG)
        pokemon_card = _card(
            "Charmander", "src-2", {"a": 1}, source_game=GameId.POKEMON
        )
        binder.create(mtg_card)
        binder.create(pokemon_card)

        assert set(binder.all_uuids(GameId.MTG)) == {mtg_card.nocab_uuid}


class TestAllCards:
    def test_returns_every_card_for_one_game(self) -> None:
        binder = CardBinder()
        mtg_card = _card("Bolt", "src-1", {"a": 1}, source_game=GameId.MTG)
        pokemon_card = _card(
            "Charmander", "src-2", {"a": 1}, source_game=GameId.POKEMON
        )
        binder.create(mtg_card)
        binder.create(pokemon_card)

        assert list(binder.all_cards(GameId.MTG)) == [mtg_card]


class TestLoad:
    def test_empty_list_returns_usable_empty_binder(self) -> None:
        binder = CardBinder.load([])

        assert binder.get_by_name(GameId.MTG, "Bolt") == []

    def test_missing_alias_ledger_file_is_not_an_error(self, tmp_path: Path) -> None:
        # A hand-written fixture with no sibling .alias_ledger.jsonl file.
        path = tmp_path / "mtg.jsonl"
        card = _card("Bolt", "src-1", {"a": 1})
        path.write_text(
            '{"nocab_uuid": "%s", "source_game": "mtg", "name": "Bolt", '
            '"raw_content": {"a": 1}, "provenance": {"data_source": "scryfall", '
            '"source_id": "src-1", "fetched_at": "2026-01-01T00:00:00+00:00"}}\n'
            % card.nocab_uuid
        )

        binder = CardBinder.load([path])

        assert binder.get_by_uuid(card.nocab_uuid) is not None

    def test_last_path_wins_on_uuid_collision_across_paths(
        self, tmp_path: Path
    ) -> None:
        card = _card("Bolt", "src-1", {"a": 1})
        first_binder = CardBinder()
        first_binder.create(card)
        first_path = tmp_path / "first.jsonl"
        first_binder.save(first_path, GameId.MTG)

        updated_card = replace(card, raw_content={"a": 999})
        second_binder = CardBinder()
        second_binder.create(updated_card)
        second_path = tmp_path / "second.jsonl"
        second_binder.save(second_path, GameId.MTG)

        merged = CardBinder.load([first_path, second_path])

        assert merged.get_by_uuid(card.nocab_uuid).raw_content == {"a": 999}


class TestSaveLoadRoundTrip:
    def test_nocab_uuid_is_bit_for_bit_identical_after_round_trip(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)
        path = tmp_path / "mtg.jsonl"

        binder.save(path, GameId.MTG)
        loaded = CardBinder.load([path])

        reloaded_card = loaded.get_by_uuid(card.nocab_uuid)
        assert reloaded_card is not None
        assert reloaded_card.nocab_uuid == card.nocab_uuid
        assert str(reloaded_card.nocab_uuid) == str(card.nocab_uuid)

    def test_writes_only_requested_games_subset(self, tmp_path: Path) -> None:
        binder = CardBinder()
        binder.create(_card("Bolt", "src-1", {"a": 1}, source_game=GameId.MTG))
        binder.create(
            _card("Charmander", "src-2", {"a": 1}, source_game=GameId.POKEMON)
        )
        path = tmp_path / "mtg.jsonl"

        binder.save(path, GameId.MTG)

        loaded = CardBinder.load([path])
        assert loaded.get_by_name(GameId.MTG, "Bolt") != []
        assert loaded.get_by_name(GameId.POKEMON, "Charmander") == []

    def test_round_trip_preserves_extra_aliases(self, tmp_path: Path) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)
        binder.register_alias(GameId.MTG, DataSource.SCRYFALL, "src-1", card.nocab_uuid)
        binder.register_alias(GameId.MTG, DataSource.ARENA, "76497", card.nocab_uuid)
        path = tmp_path / "mtg.jsonl"

        binder.save(path, GameId.MTG)
        loaded = CardBinder.load([path])

        assert (
            loaded.get_by_alias(GameId.MTG, DataSource.ARENA, "76497").nocab_uuid
            == card.nocab_uuid
        )

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        binder = CardBinder()
        binder.create(_card("Bolt", "src-1", {"a": 1}))
        nested_path = tmp_path / "does" / "not" / "exist" / "mtg.jsonl"

        binder.save(nested_path, GameId.MTG)

        assert nested_path.exists()

    def test_writes_alias_ledger_sibling_file(self, tmp_path: Path) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.create(card)
        binder.register_alias(GameId.MTG, DataSource.SCRYFALL, "src-1", card.nocab_uuid)
        path = tmp_path / "mtg.jsonl"

        binder.save(path, GameId.MTG)

        assert (tmp_path / "mtg.alias_ledger.jsonl").exists()


class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        assert CardBinder.default_output_path(
            GameId.MTG
        ) == CardBinder.DEFAULT_OUTPUT_DIR / ("mtg.jsonl")

    def test_varies_by_game(self) -> None:
        assert CardBinder.default_output_path(
            GameId.MTG
        ) != CardBinder.default_output_path(GameId.POKEMON)


class TestEnsureUnknownCard:
    def test_creates_sentinel_on_first_call(self) -> None:
        binder = CardBinder()

        unknown = binder.ensure_unknown_card(GameId.MTG)

        assert unknown.name == CardBinder.UNKNOWN_CARD_NAME
        assert unknown.source_game == GameId.MTG
        assert binder.get_by_uuid(unknown.nocab_uuid) == unknown

    def test_second_call_returns_the_same_card(self) -> None:
        binder = CardBinder()

        first = binder.ensure_unknown_card(GameId.MTG)
        second = binder.ensure_unknown_card(GameId.MTG)

        assert second == first
        assert len(list(binder.all_cards(GameId.MTG))) == 1

    def test_different_games_get_different_uuids(self) -> None:
        binder = CardBinder()

        mtg_unknown = binder.ensure_unknown_card(GameId.MTG)
        pokemon_unknown = binder.ensure_unknown_card(GameId.POKEMON)

        assert mtg_unknown.nocab_uuid != pokemon_unknown.nocab_uuid

    def test_unrelated_card_legitimately_named_unknown_is_not_mistaken_for_sentinel(
        self,
    ) -> None:
        # Regression test for the bug caught during skeleton review: the
        # short-circuit must check by this sentinel's own deterministic
        # uuid, not by name — a real ingested card happening to be named
        # "Unknown" must not be silently treated as the sentinel.
        binder = CardBinder()
        real_card_named_unknown = _card("Unknown", "src-1", {"a": 1})
        binder.create(real_card_named_unknown)

        sentinel = binder.ensure_unknown_card(GameId.MTG)

        assert sentinel.nocab_uuid != real_card_named_unknown.nocab_uuid
        assert len(list(binder.all_cards(GameId.MTG))) == 2
