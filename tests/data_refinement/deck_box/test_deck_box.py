from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _deck(
    name: str,
    card_nocab_uuids: list[UUID],
    source_game: GameId = GameId.MTG,
    provenance: Provenance | None = None,
) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=source_game,
        name=name,
        card_nocab_uuids=card_nocab_uuids,
        provenance=provenance,
    )


def _provenance(source_id: str = "deck-1") -> Provenance:
    return Provenance(
        data_source=DataSource.STS_GG,
        source_id=source_id,
        fetched_at=datetime.now(timezone.utc),
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

    def test_overrides_provenance_when_given(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()], provenance=_provenance("old"))
        box.create(deck)
        new_provenance = _provenance("new")

        updated = box.update(deck.nocab_uuid, provenance=new_provenance)

        assert updated.provenance == new_provenance
        assert box.get_by_uuid(deck.nocab_uuid).provenance == new_provenance

    def test_leaves_fields_untouched_when_not_given(self) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4(), uuid4()], provenance=_provenance())
        box.create(deck)

        updated = box.update(deck.nocab_uuid)

        assert updated.card_nocab_uuids == deck.card_nocab_uuids
        assert updated.name == deck.name
        assert updated.provenance == deck.provenance

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


class TestVersionFor:
    def test_empty_game_returns_a_fixed_digest(self) -> None:
        box = DeckBox()

        assert box.version_for(GameId.MTG) == box.version_for(GameId.MTG)
        assert isinstance(box.version_for(GameId.MTG), str)

    def test_stable_across_instances_with_the_same_content(self) -> None:
        deck = _deck("Mono Red", [uuid4(), uuid4()])
        first_box = DeckBox()
        first_box.create(replace(deck))
        second_box = DeckBox()
        second_box.create(replace(deck))

        assert first_box.version_for(GameId.MTG) == second_box.version_for(GameId.MTG)

    def test_unaffected_by_card_nocab_uuids_order(self) -> None:
        # card_nocab_uuids is a multiset (unordered, duplicates
        # meaningful) - the same deck (same nocab_uuid) with its cards
        # listed in a different order must hash identically.
        card_a, card_b = uuid4(), uuid4()
        deck = _deck("Mono Red", [card_a, card_b])
        reordered_deck = replace(deck, card_nocab_uuids=[card_b, card_a])
        first_box = DeckBox()
        first_box.create(deck)
        second_box = DeckBox()
        second_box.create(reordered_deck)

        assert first_box.version_for(GameId.MTG) == second_box.version_for(GameId.MTG)

    def test_changes_when_a_deck_is_added(self) -> None:
        box = DeckBox()
        box.create(_deck("Mono Red", [uuid4()]))
        before = box.version_for(GameId.MTG)

        box.create(_deck("Mono Blue", [uuid4()]))

        assert box.version_for(GameId.MTG) != before

    def test_unaffected_by_other_games(self) -> None:
        box = DeckBox()
        before = box.version_for(GameId.MTG)

        box.create(_deck("Fire Deck", [uuid4()], source_game=GameId.POKEMON))

        assert box.version_for(GameId.MTG) == before


class TestCardBinderVersionFor:
    def test_none_for_a_fresh_box(self) -> None:
        box = DeckBox()

        assert box.card_binder_version_for(GameId.MTG) is None

    def test_round_trips_through_save_and_load(self, tmp_path: Path) -> None:
        box = DeckBox()
        box.create(_deck("Mono Red", [uuid4()]))
        path = tmp_path / "mtg.db"

        box.save(path, GameId.MTG, "binder-v1")
        loaded = DeckBox.load([path])

        assert loaded.card_binder_version_for(GameId.MTG) == "binder-v1"

    def test_none_for_a_game_the_loaded_file_never_stamped(
        self, tmp_path: Path
    ) -> None:
        box = DeckBox()
        box.create(_deck("Fire Deck", [uuid4()], source_game=GameId.POKEMON))
        path = tmp_path / "pokemon.db"

        box.save(path, GameId.POKEMON, "binder-v1")
        loaded = DeckBox.load([path])

        assert loaded.card_binder_version_for(GameId.MTG) is None

    def test_last_path_wins_across_multiple_loaded_paths(self, tmp_path: Path) -> None:
        first_box = DeckBox()
        first_box.create(_deck("Mono Red", [uuid4()]))
        first_path = tmp_path / "first.db"
        first_box.save(first_path, GameId.MTG, "binder-v1")

        second_box = DeckBox()
        second_box.create(_deck("Mono Blue", [uuid4()]))
        second_path = tmp_path / "second.db"
        second_box.save(second_path, GameId.MTG, "binder-v2")

        loaded = DeckBox.load([first_path, second_path])

        assert loaded.card_binder_version_for(GameId.MTG) == "binder-v2"

    def test_none_after_a_mutating_call_invalidates_it(self, tmp_path: Path) -> None:
        # See DeckBox's own docstring's crash-safety section: the first
        # mutating call in a freshly-load()-ed session clears the
        # version stamp, so a crash before the next save() leaves it
        # correctly untrustworthy rather than falsely still matching.
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)
        path = tmp_path / "mtg.db"
        box.save(path, GameId.MTG, "binder-v1")

        reloaded = DeckBox.load([path])
        assert reloaded.card_binder_version_for(GameId.MTG) == "binder-v1"
        reloaded.update(deck.nocab_uuid, name="Renamed")

        assert reloaded.card_binder_version_for(GameId.MTG) is None

    def test_a_pure_read_session_never_invalidates_it(self, tmp_path: Path) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)
        path = tmp_path / "mtg.db"
        box.save(path, GameId.MTG, "binder-v1")

        reloaded = DeckBox.load([path])
        reloaded.get_by_uuid(deck.nocab_uuid)
        list(reloaded.all_decks(GameId.MTG))

        assert reloaded.card_binder_version_for(GameId.MTG) == "binder-v1"


class TestLoad:
    def test_empty_list_returns_usable_empty_box(self) -> None:
        box = DeckBox.load([])

        assert list(box.all_decks(GameId.MTG)) == []

    def test_single_path_connects_directly_even_when_missing(
        self, tmp_path: Path
    ) -> None:
        # The important correction from this rewrite's second design
        # pass: a brand-new path (never saved before) must still get a
        # DIRECT connection, not a throwaway :memory: box - otherwise a
        # first-ever ingestion's progress would be lost on a crash,
        # never reaching save(). See DeckBox's own docstring.
        path = tmp_path / "does" / "not" / "exist" / "mtg.db"

        box = DeckBox.load([path])
        box.create(_deck("Mono Red", [uuid4()]))

        assert path.exists()
        reopened = DeckBox.load([path])
        assert len(list(reopened.all_decks(GameId.MTG))) == 1

    def test_last_path_wins_on_uuid_collision_across_paths(
        self, tmp_path: Path
    ) -> None:
        deck = _deck("Mono Red", [uuid4()])
        first_box = DeckBox()
        first_box.create(deck)
        first_path = tmp_path / "first.db"
        first_box.save(first_path, GameId.MTG, "binder-v1")

        updated_deck = replace(deck, card_nocab_uuids=[uuid4(), uuid4()])
        second_box = DeckBox()
        second_box.create(updated_deck)
        second_path = tmp_path / "second.db"
        second_box.save(second_path, GameId.MTG, "binder-v2")

        merged = DeckBox.load([first_path, second_path])

        assert merged.get_by_uuid(deck.nocab_uuid).card_nocab_uuids == (
            updated_deck.card_nocab_uuids
        )

    def test_multi_path_load_never_mutates_any_given_path(self, tmp_path: Path) -> None:
        first_box = DeckBox()
        first_box.create(_deck("Mono Red", [uuid4()]))
        first_path = tmp_path / "first.db"
        first_box.save(first_path, GameId.MTG, "binder-v1")
        first_mtime = first_path.stat().st_mtime
        first_size = first_path.stat().st_size

        second_box = DeckBox()
        second_box.create(_deck("Mono Blue", [uuid4()]))
        second_path = tmp_path / "second.db"
        second_box.save(second_path, GameId.MTG, "binder-v2")

        merged = DeckBox.load([first_path, second_path])
        merged.create(_deck("Newly created only in the merged box", [uuid4()]))

        assert first_path.stat().st_mtime == first_mtime
        assert first_path.stat().st_size == first_size
        # Neither original file gained the deck created only on the
        # merged (fresh :memory:) box - confirms the merge is a real
        # copy, not a live connection to either input.
        assert len(list(DeckBox.load([first_path]).all_decks(GameId.MTG))) == 1
        assert len(list(DeckBox.load([second_path]).all_decks(GameId.MTG))) == 1


class TestSaveLoadRoundTrip:
    def test_nocab_uuid_is_bit_for_bit_identical_after_round_trip(
        self, tmp_path: Path
    ) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()])
        box.create(deck)
        path = tmp_path / "mtg.db"

        box.save(path, GameId.MTG, "binder-v1")
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
        path = tmp_path / "mtg.db"

        box.save(path, GameId.MTG, "binder-v1")
        loaded = DeckBox.load([path])

        reloaded_deck = loaded.get_by_uuid(deck.nocab_uuid)
        assert sorted(reloaded_deck.card_nocab_uuids, key=str) == sorted(
            deck.card_nocab_uuids, key=str
        )
        assert reloaded_deck.card_nocab_uuids.count(repeated) == 3

    def test_backup_copies_every_game_regardless_of_source_game_argument(
        self, tmp_path: Path
    ) -> None:
        # Unlike the old JSONL implementation, save() no longer filters
        # by source_game when it has to copy a still-":memory:" box's
        # full state to path (the not-yet-connected branch) - see
        # save()'s own docstring. source_game is purely the metadata
        # stamp's key here, not a row filter.
        box = DeckBox()
        box.create(_deck("Mono Red", [uuid4()], source_game=GameId.MTG))
        box.create(_deck("Fire Deck", [uuid4()], source_game=GameId.POKEMON))
        path = tmp_path / "mtg.db"

        box.save(path, GameId.MTG, "binder-v1")

        loaded = DeckBox.load([path])
        assert list(loaded.all_decks(GameId.MTG)) != []
        assert list(loaded.all_decks(GameId.POKEMON)) != []

    def test_preserves_provenance_when_present(self, tmp_path: Path) -> None:
        box = DeckBox()
        provenance = _provenance("run-1")
        deck = _deck("Mono Red", [uuid4()], provenance=provenance)
        box.create(deck)
        path = tmp_path / "mtg.db"

        box.save(path, GameId.MTG, "binder-v1")
        loaded = DeckBox.load([path])

        reloaded_deck = loaded.get_by_uuid(deck.nocab_uuid)
        assert reloaded_deck.provenance == provenance

    def test_provenance_defaults_to_none_when_absent(self, tmp_path: Path) -> None:
        box = DeckBox()
        deck = _deck("Mono Red", [uuid4()], provenance=None)
        box.create(deck)
        path = tmp_path / "mtg.db"

        box.save(path, GameId.MTG, "binder-v1")
        loaded = DeckBox.load([path])

        assert loaded.get_by_uuid(deck.nocab_uuid).provenance is None

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        box = DeckBox()
        box.create(_deck("Mono Red", [uuid4()]))
        nested_path = tmp_path / "does" / "not" / "exist" / "mtg.db"

        box.save(nested_path, GameId.MTG, "binder-v1")

        assert nested_path.exists()


class TestUuidsRankedRandomly:
    def test_ranks_are_a_dense_one_to_total_permutation(self) -> None:
        box = DeckBox()
        uuids = [uuid4() for _ in range(20)]
        for nocab_uuid in uuids:
            box.create(
                GenericDeck(
                    nocab_uuid=nocab_uuid,
                    source_game=GameId.MTG,
                    name="d",
                    card_nocab_uuids=[],
                )
            )

        ranked = list(box.uuids_ranked_randomly(GameId.MTG, seed=1))

        assert {nocab_uuid for nocab_uuid, _, _ in ranked} == set(uuids)
        assert sorted(rank for _, rank, _ in ranked) == list(range(1, 21))
        assert all(total == 20 for _, _, total in ranked)

    def test_reproducible_given_the_same_seed(self) -> None:
        box = DeckBox()
        for _ in range(15):
            box.create(_deck("d", [uuid4()]))

        first = sorted(
            box.uuids_ranked_randomly(GameId.MTG, seed=42), key=lambda t: t[1]
        )
        second = sorted(
            box.uuids_ranked_randomly(GameId.MTG, seed=42), key=lambda t: t[1]
        )

        assert first == second

    def test_different_seeds_almost_certainly_differ(self) -> None:
        box = DeckBox()
        for _ in range(15):
            box.create(_deck("d", [uuid4()]))

        first = sorted(
            box.uuids_ranked_randomly(GameId.MTG, seed=1), key=lambda t: t[1]
        )
        second = sorted(
            box.uuids_ranked_randomly(GameId.MTG, seed=2), key=lambda t: t[1]
        )

        assert first != second

    def test_empty_game_yields_nothing(self) -> None:
        box = DeckBox()

        assert list(box.uuids_ranked_randomly(GameId.MTG, seed=1)) == []


class TestMutatingMethodsAreAtomic:
    def test_a_crash_between_the_deck_row_and_its_card_rows_commits_nothing(
        self, tmp_path: Path
    ) -> None:
        # create() writes its deck row and its deck_cards rows, then
        # commits ONCE at the end - not once per write - specifically
        # so a crash between them can never leave a deck row with no
        # cards. Simulate that crash directly: perform the row write
        # create() does internally, without its own method's final
        # commit, then "crash" (close without committing).
        path = tmp_path / "mtg.db"
        box = DeckBox.load([path])
        deck = _deck("Mono Red", [uuid4(), uuid4()])

        box._tables.insert_deck_row(
            deck
        )  # what create() does before its cards write + commit
        box._connection.close()  # never committed

        reopened = DeckBox.load([path])
        assert (
            reopened.get_by_uuid(deck.nocab_uuid) is None
        ), "an uncommitted partial write must not be visible after reconnecting"

    def test_a_caught_exception_mid_create_never_leaks_into_a_later_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The failure mode the process-crash test above does NOT cover:
        # a Python exception (not a process crash) raised mid-create(),
        # caught by a caller who keeps reusing the SAME DeckBox instance
        # for a later, unrelated create() - real callers loop
        # create()/create_if_absent() per row against one shared box
        # (e.g. every DeckExtractionStage). Without `with
        # self._connection:`'s rollback, the doomed call's deck row
        # would already be durably committed the moment
        # insert_deck_cards raises, so it would survive even
        # though the whole create() call never returned successfully -
        # and would still be there after a later, unrelated call.
        path = tmp_path / "mtg.db"
        box = DeckBox.load([path])
        doomed_deck = _deck("Doomed", [uuid4()])

        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated mid-create failure")

        monkeypatch.setattr(box._tables, "insert_deck_cards", _raise)
        with pytest.raises(RuntimeError, match="simulated mid-create failure"):
            box.create(doomed_deck)

        # Immediately after the caught exception - before any further
        # calls - nothing from the doomed call should be visible.
        assert box.get_by_uuid(doomed_deck.nocab_uuid) is None

        # Reuse the SAME instance for an unrelated, successful create()
        # - the doomed row must not have been silently folded into it.
        monkeypatch.undo()
        survivor_deck = _deck("Survivor", [uuid4()])
        box.create(survivor_deck)

        assert box.get_by_uuid(doomed_deck.nocab_uuid) is None, (
            "the doomed call's partial write must not surface even after "
            "this same instance went on to commit a later call successfully"
        )
        assert box.get_by_uuid(survivor_deck.nocab_uuid) is not None


class TestCrashDuringExtractionThenRetry:
    def test_retry_after_a_partial_run_converges_without_duplicating(
        self, tmp_path: Path
    ) -> None:
        # Simulates the idempotent get_by_uuid()-then-create()-or-update()
        # pattern every DeckExtractionStage follows (see extraction.py's
        # own docstring) - a "crash" here is just abandoning a box after
        # some rows without ever calling save().
        path = tmp_path / "mtg.db"
        all_uuids = [uuid4() for _ in range(5)]

        def _extract_idempotently(box: DeckBox, uuids: list[UUID]) -> None:
            for nocab_uuid in uuids:
                if box.get_by_uuid(nocab_uuid) is None:
                    box.create(
                        GenericDeck(
                            nocab_uuid=nocab_uuid,
                            source_game=GameId.MTG,
                            name="d",
                            card_nocab_uuids=[],
                        )
                    )

        # First (partial) run: only processes the first 3 of 5 rows,
        # then "crashes" - never reaches save().
        first_run = DeckBox.load([path])
        _extract_idempotently(first_run, all_uuids[:3])

        assert path.exists()  # already durable, per the direct-connect design
        assert DeckBox.load([path]).card_binder_version_for(GameId.MTG) is None

        # Retry: a fresh process re-runs extraction from the start over
        # the SAME full raw source (all 5 uuids), then saves.
        retry_run = DeckBox.load([path])
        _extract_idempotently(retry_run, all_uuids)
        retry_run.save(path, GameId.MTG, "binder-v2")

        final = DeckBox.load([path])
        assert set(final.all_uuids(GameId.MTG)) == set(all_uuids)
        assert final.card_binder_version_for(GameId.MTG) == "binder-v2"


class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        assert DeckBox.default_output_path(GameId.MTG) == DeckBox.DEFAULT_OUTPUT_DIR / (
            "mtg.db"
        )

    def test_varies_by_game(self) -> None:
        assert DeckBox.default_output_path(GameId.MTG) != DeckBox.default_output_path(
            GameId.POKEMON
        )
