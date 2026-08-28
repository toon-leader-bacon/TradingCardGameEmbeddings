from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import AddOutcome, CardBinder
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


class TestAdd:
    def test_first_candidate_for_a_name_is_inserted(self) -> None:
        binder = CardBinder()
        candidate = _card("Bolt", "src-1", {"a": 1})

        result = binder.add(candidate)

        assert result.outcome == AddOutcome.INSERTED
        assert result.stored_card == candidate

    def test_strictly_richer_candidate_updates_content(self) -> None:
        binder = CardBinder()
        first = _card("Bolt", "src-1", {"a": 1})
        binder.add(first)
        richer = _card("Bolt", "src-2", {"a": 1, "b": 2})

        result = binder.add(richer)

        assert result.outcome == AddOutcome.CONTENT_UPDATED
        assert result.stored_card.raw_content == richer.raw_content
        assert result.stored_card.provenance == richer.provenance

    def test_content_update_preserves_existing_nocab_uuid(self) -> None:
        binder = CardBinder()
        first = _card("Bolt", "src-1", {"a": 1})
        binder.add(first)
        richer = _card("Bolt", "src-2", {"a": 1, "b": 2})

        result = binder.add(richer)

        assert result.stored_card.nocab_uuid == first.nocab_uuid
        assert result.stored_card.nocab_uuid != richer.nocab_uuid

    def test_less_rich_candidate_keeps_existing(self) -> None:
        binder = CardBinder()
        richer = _card("Bolt", "src-1", {"a": 1, "b": 2})
        binder.add(richer)
        poorer = _card("Bolt", "src-2", {"a": 1})

        result = binder.add(poorer)

        assert result.outcome == AddOutcome.KEPT_EXISTING
        assert result.stored_card.raw_content == richer.raw_content
        assert result.stored_card.nocab_uuid == richer.nocab_uuid

    def test_exact_tie_keeps_existing(self) -> None:
        binder = CardBinder()
        first = _card("Bolt", "src-1", {"a": 1, "b": 2})
        binder.add(first)
        tied = _card("Bolt", "src-2", {"a": 1, "b": 2})

        result = binder.add(tied)

        assert result.outcome == AddOutcome.KEPT_EXISTING
        assert result.stored_card.nocab_uuid == first.nocab_uuid

    def test_losing_and_winning_aliases_both_resolve_to_survivor(self) -> None:
        binder = CardBinder()
        first = _card("Bolt", "src-1", {"a": 1})
        binder.add(first)
        richer = _card("Bolt", "src-2", {"a": 1, "b": 2})
        binder.add(richer)

        assert (
            binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-1").nocab_uuid
            == first.nocab_uuid
        )
        assert (
            binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-2").nocab_uuid
            == first.nocab_uuid
        )

    def test_different_names_do_not_collide(self) -> None:
        binder = CardBinder()
        bolt = _card("Bolt", "src-1", {"a": 1})
        shock = _card("Shock", "src-2", {"a": 1})

        binder.add(bolt)
        result = binder.add(shock)

        assert result.outcome == AddOutcome.INSERTED
        assert binder.get_by_name(GameId.MTG, "Bolt").nocab_uuid == bolt.nocab_uuid
        assert binder.get_by_name(GameId.MTG, "Shock").nocab_uuid == shock.nocab_uuid

    def test_same_name_different_game_does_not_collide(self) -> None:
        binder = CardBinder()
        mtg_card = _card("Boost", "src-1", {"a": 1}, source_game=GameId.MTG)
        pokemon_card = _card("Boost", "src-2", {"a": 1}, source_game=GameId.POKEMON)

        binder.add(mtg_card)
        result = binder.add(pokemon_card)

        assert result.outcome == AddOutcome.INSERTED


class TestGetByUuid:
    def test_found(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.add(card)

        assert binder.get_by_uuid(card.nocab_uuid) == card

    def test_not_found(self) -> None:
        binder = CardBinder()

        assert binder.get_by_uuid(uuid4()) is None


class TestGetByAlias:
    def test_found(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.add(card)

        assert binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-1") == card

    def test_not_found(self) -> None:
        binder = CardBinder()

        assert binder.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "nonexistent") is None

    def test_wrong_game_not_found(self) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt", "src-1", {"a": 1}, source_game=GameId.MTG))

        assert binder.get_by_alias(GameId.POKEMON, DataSource.SCRYFALL, "src-1") is None

    def test_wrong_data_source_not_found(self) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt", "src-1", {"a": 1}, data_source=DataSource.SCRYFALL))

        assert binder.get_by_alias(GameId.MTG, DataSource.ARENA, "src-1") is None

    def test_extra_alias_resolves_via_register_alias(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.add(card)

        binder.register_alias(GameId.MTG, DataSource.ARENA, "76497", card.nocab_uuid)

        assert binder.get_by_alias(GameId.MTG, DataSource.ARENA, "76497") == card


class TestRegisterAlias:
    def test_raises_for_unknown_nocab_uuid(self) -> None:
        binder = CardBinder()

        with pytest.raises(ValueError):
            binder.register_alias(GameId.MTG, DataSource.ARENA, "76497", uuid4())


class TestGetByName:
    def test_found(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.add(card)

        assert binder.get_by_name(GameId.MTG, "Bolt") == card

    def test_not_found(self) -> None:
        binder = CardBinder()

        assert binder.get_by_name(GameId.MTG, "Nonexistent") is None


class TestGetByNameRegex:
    def test_exact_name_as_pattern_matches(self) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.add(card)

        assert binder.get_by_name_regex(GameId.MTG, "^Bolt$") == [card]

    def test_mdfc_fallback_pattern_matches_combined_name(self) -> None:
        binder = CardBinder()
        card = _card("Bruce Banner // The Incredible Hulk", "src-1", {"a": 1})
        binder.add(card)

        matches = binder.get_by_name_regex(GameId.MTG, "^Bruce Banner( //.*)?$")

        assert matches == [card]

    def test_no_match_returns_empty_list(self) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt", "src-1", {"a": 1}))

        assert binder.get_by_name_regex(GameId.MTG, "^Shock$") == []

    def test_multiple_matches_all_returned(self) -> None:
        binder = CardBinder()
        bolt = _card("Bolt", "src-1", {"a": 1})
        bolts = _card("Bolts", "src-2", {"a": 1})
        binder.add(bolt)
        binder.add(bolts)

        matches = binder.get_by_name_regex(GameId.MTG, "^Bolt")

        assert set(c.nocab_uuid for c in matches) == {
            bolt.nocab_uuid,
            bolts.nocab_uuid,
        }

    def test_respects_source_game_filtering(self) -> None:
        binder = CardBinder()
        binder.add(_card("Boost", "src-1", {"a": 1}, source_game=GameId.MTG))

        assert binder.get_by_name_regex(GameId.POKEMON, "^Boost$") == []


class TestLoad:
    def test_empty_list_returns_usable_empty_binder(self) -> None:
        binder = CardBinder.load([])

        assert binder.get_by_name(GameId.MTG, "Bolt") is None

    def test_single_file_round_trips(self, tmp_path: Path) -> None:
        original = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        original.add(card)
        path = tmp_path / "mtg.jsonl"
        original.save(path, GameId.MTG)

        loaded = CardBinder.load([path])

        got = loaded.get_by_name(GameId.MTG, "Bolt")
        assert got == card

    def test_merges_multiple_files_via_add_collision_logic(self, tmp_path: Path) -> None:
        first_binder = CardBinder()
        first_card = _card("Bolt", "src-1", {"a": 1})
        first_binder.add(first_card)
        first_path = tmp_path / "first.jsonl"
        first_binder.save(first_path, GameId.MTG)

        second_binder = CardBinder()
        richer_card = _card("Bolt", "src-2", {"a": 1, "b": 2})
        second_binder.add(richer_card)
        second_path = tmp_path / "second.jsonl"
        second_binder.save(second_path, GameId.MTG)

        merged = CardBinder.load([first_path, second_path])

        got = merged.get_by_name(GameId.MTG, "Bolt")
        assert got.raw_content == richer_card.raw_content
        assert got.nocab_uuid == first_card.nocab_uuid
        assert (
            merged.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-1").nocab_uuid
            == first_card.nocab_uuid
        )
        assert (
            merged.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-2").nocab_uuid
            == first_card.nocab_uuid
        )

    def test_aliases_survive_when_their_own_file_loses_a_cross_file_merge(
        self, tmp_path: Path
    ) -> None:
        # first.jsonl is loaded first, so its card's nocab_uuid becomes
        # canonical for "Bolt" in the merged binder. second.jsonl
        # already resolved its OWN internal collision before being
        # saved (two source_ids -> one uuid, a genuine alias pair) —
        # but that whole card then loses the cross-file merge against
        # first.jsonl's card, so second.jsonl's own nocab_uuid never
        # becomes canonical anywhere in the merged binder. Both of its
        # aliases must still resolve, to the MERGED uuid (first's), not
        # silently vanish because their file's own uuid no longer
        # exists as a key anywhere in the merged binder.
        first_binder = CardBinder()
        first_card = _card("Bolt", "src-a1", {"a": 1})
        first_binder.add(first_card)
        first_path = tmp_path / "first.jsonl"
        first_binder.save(first_path, GameId.MTG)

        second_binder = CardBinder()
        second_binder.add(_card("Bolt", "src-b1", {"a": 1, "b": 2}))
        second_binder.add(_card("Bolt", "src-b2", {"a": 1, "b": 2, "c": 3}))
        second_path = tmp_path / "second.jsonl"
        second_binder.save(second_path, GameId.MTG)

        merged = CardBinder.load([first_path, second_path])

        got = merged.get_by_name(GameId.MTG, "Bolt")
        assert got.nocab_uuid == first_card.nocab_uuid
        assert got.raw_content == {"a": 1, "b": 2, "c": 3}
        assert (
            merged.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-b1").nocab_uuid
            == first_card.nocab_uuid
        )
        assert (
            merged.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-b2").nocab_uuid
            == first_card.nocab_uuid
        )

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

        assert binder.get_by_name(GameId.MTG, "Bolt").nocab_uuid == card.nocab_uuid

    def test_multiple_extra_aliases_all_survive_round_trip(self, tmp_path: Path) -> None:
        binder = CardBinder()
        card = _card("Bolt", "src-1", {"a": 1})
        binder.add(card)
        binder.register_alias(GameId.MTG, DataSource.ARENA, "76497", card.nocab_uuid)
        binder.register_alias(GameId.MTG, DataSource.MTGO, "88685", card.nocab_uuid)
        path = tmp_path / "mtg.jsonl"

        binder.save(path, GameId.MTG)
        loaded = CardBinder.load([path])

        assert (
            loaded.get_by_alias(GameId.MTG, DataSource.ARENA, "76497").nocab_uuid
            == card.nocab_uuid
        )
        assert (
            loaded.get_by_alias(GameId.MTG, DataSource.MTGO, "88685").nocab_uuid
            == card.nocab_uuid
        )


class TestSave:
    def test_writes_only_requested_games_subset(self, tmp_path: Path) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt", "src-1", {"a": 1}, source_game=GameId.MTG))
        binder.add(_card("Charmander", "src-2", {"a": 1}, source_game=GameId.POKEMON))
        path = tmp_path / "mtg.jsonl"

        binder.save(path, GameId.MTG)

        loaded = CardBinder.load([path])
        assert loaded.get_by_name(GameId.MTG, "Bolt") is not None
        assert loaded.get_by_name(GameId.POKEMON, "Charmander") is None

    def test_round_trip_preserves_losing_and_winning_aliases(self, tmp_path: Path) -> None:
        binder = CardBinder()
        first = _card("Bolt", "src-1", {"a": 1})
        binder.add(first)
        richer = _card("Bolt", "src-2", {"a": 1, "b": 2})
        binder.add(richer)
        path = tmp_path / "mtg.jsonl"

        binder.save(path, GameId.MTG)
        loaded = CardBinder.load([path])

        assert (
            loaded.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-1").nocab_uuid
            == first.nocab_uuid
        )
        assert (
            loaded.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-2").nocab_uuid
            == first.nocab_uuid
        )

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt", "src-1", {"a": 1}))
        nested_path = tmp_path / "does" / "not" / "exist" / "mtg.jsonl"

        binder.save(nested_path, GameId.MTG)

        assert nested_path.exists()

    def test_writes_alias_ledger_sibling_file(self, tmp_path: Path) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt", "src-1", {"a": 1}))
        path = tmp_path / "mtg.jsonl"

        binder.save(path, GameId.MTG)

        assert (tmp_path / "mtg.alias_ledger.jsonl").exists()
