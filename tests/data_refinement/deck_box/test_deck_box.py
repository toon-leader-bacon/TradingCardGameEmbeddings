from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericDeck
from src.schema.game_id import GameId


def _deck(
    name: str,
    card_nocab_uuids: list[UUID],
    source_game: GameId = GameId.MTG,
) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=source_game,
        name=name,
        card_nocab_uuids=card_nocab_uuids,
    )


class TestCreate:
    def test_inserts_new_deck(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4(), uuid4()])

        result = box.create(deck)

        assert result == deck
        assert box.get_by_uuid(deck.nocab_uuid) == deck

    def test_raises_on_duplicate_uuid(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)

        with pytest.raises(ValueError):
            box.create(deck)


class TestCreateIfAbsent:
    def test_inserts_new_deck(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4(), uuid4()])

        result = box.create_if_absent(deck)

        assert result == deck
        assert box.get_by_uuid(deck.nocab_uuid) == deck

    def test_recurrence_is_a_no_op_and_returns_existing(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create_if_absent(deck)
        recurrence = _deck("Different Name", [uuid4(), uuid4()])
        recurrence = replace(recurrence, nocab_uuid=deck.nocab_uuid)

        result = box.create_if_absent(recurrence)

        assert result == deck
        assert box.get_by_uuid(deck.nocab_uuid) == deck


class TestUpdate:
    def test_overrides_card_nocab_uuids_when_given(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)
        new_cards = [uuid4(), uuid4(), uuid4()]

        updated = box.update(deck.nocab_uuid, card_nocab_uuids=new_cards)

        assert updated.card_nocab_uuids == new_cards
        assert updated.nocab_uuid == deck.nocab_uuid

    def test_overrides_name_when_given(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)

        updated = box.update(deck.nocab_uuid, name="Burn")

        assert updated.name == "Burn"
        assert box.get_by_uuid(deck.nocab_uuid).name == "Burn"

    def test_leaves_fields_untouched_when_not_given(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4(), uuid4()])
        box.create(deck)

        updated = box.update(deck.nocab_uuid)

        assert updated.card_nocab_uuids == deck.card_nocab_uuids
        assert updated.name == deck.name

    def test_card_nocab_uuids_replaced_wholesale_not_merged(self) -> None:
        box = DeckBox()
        shared = uuid4()
        deck = _deck("Mono Red", [shared, uuid4()])
        box.create(deck)
        replacement_cards = [shared]

        updated = box.update(deck.nocab_uuid, card_nocab_uuids=replacement_cards)

        assert updated.card_nocab_uuids == [shared]

    def test_raises_if_uuid_not_stored(self) -> None:
        box = DeckBox()

        with pytest.raises(KeyError):
            box.update(uuid4(), name="Nonexistent")


class TestReplace:
    def test_fully_swaps_content(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)
        replacement = GenericDeck(
            nocab_uuid=deck.nocab_uuid,
            source_game=GameId.MTG,
            name="Burn",
            card_nocab_uuids=[uuid4(), uuid4()],
        )

        result = box.replace(deck.nocab_uuid, replacement)

        assert result == replacement
        assert box.get_by_uuid(deck.nocab_uuid) == replacement

    def test_raises_on_uuid_mismatch(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)
        mismatched = _deck("Other Deck", [uuid4()])

        with pytest.raises(ValueError):
            box.replace(deck.nocab_uuid, mismatched)

    def test_raises_if_uuid_not_stored(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])

        with pytest.raises(KeyError):
            box.replace(deck.nocab_uuid, deck)


class TestDelete:
    def test_removes_deck(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)

        box.delete(deck.nocab_uuid)

        assert box.get_by_uuid(deck.nocab_uuid) is None

    def test_raises_if_uuid_not_stored(self) -> None:
        box = DeckBox()

        with pytest.raises(KeyError):
            box.delete(uuid4())


class TestGetByUuid:
    def test_found(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)

        assert box.get_by_uuid(deck.nocab_uuid) == deck

    def test_not_found(self) -> None:
        box = DeckBox()

        assert box.get_by_uuid(uuid4()) is None


class TestAllUuids:
    def test_no_filter_returns_every_uuid(self) -> None:
        box = DeckBox()
        mtg_deck = _deck("Mono Red", [uuid4()], source_game=GameId.MTG)
        pokemon_deck = _deck("Fire Deck", [uuid4()], source_game=GameId.POKEMON)
        box.create(mtg_deck)
        box.create(pokemon_deck)

        assert set(box.all_uuids()) == {mtg_deck.nocab_uuid, pokemon_deck.nocab_uuid}

    def test_filter_by_game(self) -> None:
        box = DeckBox()
        mtg_deck = _deck("Mono Red", [uuid4()], source_game=GameId.MTG)
        pokemon_deck = _deck("Fire Deck", [uuid4()], source_game=GameId.POKEMON)
        box.create(mtg_deck)
        box.create(pokemon_deck)

        assert set(box.all_uuids(GameId.MTG)) == {mtg_deck.nocab_uuid}


class TestAllDecks:
    def test_returns_every_deck_for_one_game(self) -> None:
        box = DeckBox()
        mtg_deck = _deck("Mono Red", [uuid4()], source_game=GameId.MTG)
        pokemon_deck = _deck("Fire Deck", [uuid4()], source_game=GameId.POKEMON)
        box.create(mtg_deck)
        box.create(pokemon_deck)

        assert list(box.all_decks(GameId.MTG)) == [mtg_deck]

    def test_empty_game_returns_empty_list(self) -> None:
        box = DeckBox()

        assert list(box.all_decks(GameId.MTG)) == []


class TestLoad:
    def test_empty_list_returns_usable_empty_box(self) -> None:
        box = DeckBox.load([])

        assert list(box.all_decks(GameId.MTG)) == []

    def test_last_path_wins_on_uuid_collision_across_paths(
        self, tmp_path: Path
    ) -> None:
        deck = _deck("Mono Red", [uuid4()])
        first_box = DeckBox()
        first_box.create(deck)
        first_path = tmp_path / "first.jsonl"
        first_box.save(first_path, GameId.MTG)

        updated_deck = replace(deck, card_nocab_uuids=[uuid4(), uuid4()])
        second_box = DeckBox()
        second_box.create(updated_deck)
        second_path = tmp_path / "second.jsonl"
        second_box.save(second_path, GameId.MTG)

        merged = DeckBox.load([first_path, second_path])

        assert merged.get_by_uuid(deck.nocab_uuid).card_nocab_uuids == (
            updated_deck.card_nocab_uuids
        )


class TestSaveLoadRoundTrip:
    def test_nocab_uuid_is_bit_for_bit_identical_after_round_trip(
        self, tmp_path: Path
    ) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)
        path = tmp_path / "mtg.jsonl"

        box.save(path, GameId.MTG)
        loaded = DeckBox.load([path])

        reloaded_deck = loaded.get_by_uuid(deck.nocab_uuid)
        assert reloaded_deck is not None
        assert reloaded_deck.nocab_uuid == deck.nocab_uuid
        assert str(reloaded_deck.nocab_uuid) == str(deck.nocab_uuid)

    def test_preserves_duplicate_card_uuids(self, tmp_path: Path) -> None:
        # card_nocab_uuids is a multiset: a repeated uuid represents
        # multiple copies of the same card and must survive round-trip
        # exactly, not collapse to a unique set.
        repeated = uuid4()
        box = DeckBox()
        deck = _deck("Mono Red", [repeated, repeated, repeated, uuid4()])
        box.create(deck)
        path = tmp_path / "mtg.jsonl"

        box.save(path, GameId.MTG)
        loaded = DeckBox.load([path])

        reloaded_deck = loaded.get_by_uuid(deck.nocab_uuid)
        assert sorted(reloaded_deck.card_nocab_uuids, key=str) == sorted(
            deck.card_nocab_uuids, key=str
        )
        assert reloaded_deck.card_nocab_uuids.count(repeated) == 3

    def test_writes_only_requested_games_subset(self, tmp_path: Path) -> None:
        box = DeckBox()
        box.create(_deck("Mono Red", [uuid4()], source_game=GameId.MTG))
        box.create(_deck("Fire Deck", [uuid4()], source_game=GameId.POKEMON))
        path = tmp_path / "mtg.jsonl"

        box.save(path, GameId.MTG)

        loaded = DeckBox.load([path])
        assert list(loaded.all_decks(GameId.MTG)) != []
        assert list(loaded.all_decks(GameId.POKEMON)) == []

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        box = DeckBox()
        box.create(_deck("Mono Red", [uuid4()]))
        nested_path = tmp_path / "does" / "not" / "exist" / "mtg.jsonl"

        box.save(nested_path, GameId.MTG)

        assert nested_path.exists()


class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        assert DeckBox.default_output_path(GameId.MTG) == DeckBox.DEFAULT_OUTPUT_DIR / (
            "mtg.jsonl"
        )

    def test_varies_by_game(self) -> None:
        assert DeckBox.default_output_path(GameId.MTG) != DeckBox.default_output_path(
            GameId.POKEMON
        )
