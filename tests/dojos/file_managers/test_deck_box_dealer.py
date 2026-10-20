import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.file_managers.deck_box_dealer import DeckBoxDealer
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec, UnknownDeckSplitError
from src.schema.splits import Split


def _deck(game: GameId = GameId.MTG) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=game,
        name="deck",
        card_nocab_uuids=[uuid4()],
        provenance=None,
    )


def _box_with_decks(count: int) -> tuple[DeckBox, list[UUID]]:
    box = DeckBox()
    uuids = []
    for _ in range(count):
        deck = _deck()
        box.create(deck)
        uuids.append(deck.nocab_uuid)
    return box, uuids


def _holdout(
    deck_split_ratios: tuple[float, float, float] | None = (8, 1, 1),
    seed: int = 1,
    held_out_games: frozenset[GameId] = frozenset(),
) -> HoldoutSpec:
    return HoldoutSpec(
        seed=seed,
        tier_ratios=(1, 0, 0),
        held_out_games=held_out_games,
        deck_split_ratios=deck_split_ratios,
    )


def _dealer(
    box: DeckBox,
    tmp_path: Path,
    holdout: HoldoutSpec | None = None,
    name: str = "dealer",
) -> DeckBoxDealer:
    return DeckBoxDealer(
        box, GameId.MTG, tmp_path / f"{name}.db", holdout or _holdout()
    )


def _dealt_uuids(dealer: DeckBoxDealer, split: Split) -> list[UUID]:
    return [
        deck.nocab_uuid
        for group in dealer.decks_for(split, 1, shuffle=False)
        for deck in group
    ]


def _split_counts(dealer: DeckBoxDealer) -> tuple[int, int, int]:
    return (
        dealer.deck_count(Split.TRAIN),
        dealer.deck_count(Split.TEST),
        dealer.deck_count(Split.VALIDATION),
    )


class TestSourceGameAndCardBinderVersion:
    def test_source_game_is_the_constructor_argument(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(1)

        dealer = _dealer(box, tmp_path)

        assert dealer.source_game == GameId.MTG

    def test_card_binder_version_is_none_for_an_unstamped_box(
        self, tmp_path: Path
    ) -> None:
        box, _ = _box_with_decks(1)

        dealer = _dealer(box, tmp_path)

        assert dealer.card_binder_version is None

    def test_card_binder_version_reflects_the_box(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(1)
        path = tmp_path / "mtg.db"
        box.save(path, GameId.MTG, "binder-v1")
        loaded = DeckBox.load([path])

        dealer = _dealer(loaded, tmp_path)

        assert dealer.card_binder_version == "binder-v1"

    def test_deck_split_key_is_the_holdouts_for_this_game(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(1)
        holdout = _holdout()

        dealer = _dealer(box, tmp_path, holdout)

        assert dealer.deck_split_key == holdout.deck_split_key(GameId.MTG)


class TestInit:
    def test_an_unknown_deck_split_raises_before_touching_disk(
        self, tmp_path: Path
    ) -> None:
        box, _ = _box_with_decks(1)

        with pytest.raises(UnknownDeckSplitError):
            _dealer(box, tmp_path, _holdout(deck_split_ratios=None))
        assert not (tmp_path / "dealer.db").exists()

    def test_only_considers_the_given_game(self, tmp_path: Path) -> None:
        box = DeckBox()
        mtg_deck = _deck()
        box.create(mtg_deck)
        box.create(_deck(GameId.POKEMON))

        dealer = _dealer(box, tmp_path)

        dealt = {uuid for split in Split for uuid in _dealt_uuids(dealer, split)}
        assert dealt == {mtg_deck.nocab_uuid}

    def test_a_current_index_is_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        box, _ = _box_with_decks(10)
        _dealer(box, tmp_path)

        def _fail(self: DeckBoxDealer, deck_set: str) -> None:
            raise AssertionError("a current index must not be rebuilt")

        monkeypatch.setattr(DeckBoxDealer, "_assign_splits", _fail)
        reused = _dealer(box, tmp_path)

        assert sum(_split_counts(reused)) == 10

    def test_a_changed_deck_split_rebuilds_the_index(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(10)
        _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 0, 0)))

        rebuilt = _dealer(box, tmp_path, _holdout(deck_split_ratios=(0, 1, 0)))

        assert _split_counts(rebuilt) == (0, 10, 0)

    def test_a_changed_tier_split_alone_keeps_the_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        box, _ = _box_with_decks(10)
        _dealer(box, tmp_path)
        other_tiers = HoldoutSpec(
            seed=1,
            tier_ratios=(8, 1, 1),
            held_out_games=frozenset(),
            deck_split_ratios=(8, 1, 1),
        )

        def _fail(self: DeckBoxDealer, deck_set: str) -> None:
            raise AssertionError("card tiers do not change a deck split")

        monkeypatch.setattr(DeckBoxDealer, "_assign_splits", _fail)
        _dealer(box, tmp_path, other_tiers)

    def test_a_grown_box_is_reindexed_and_old_decks_keep_their_split(
        self, tmp_path: Path
    ) -> None:
        box, _ = _box_with_decks(20)
        first = _dealer(box, tmp_path)
        before = {split: set(_dealt_uuids(first, split)) for split in Split}
        new_deck = _deck()
        box.create(new_deck)

        second = _dealer(box, tmp_path)

        after = {split: set(_dealt_uuids(second, split)) for split in Split}
        assert sum(_split_counts(second)) == 21
        for split in Split:
            assert after[split] - {new_deck.nocab_uuid} == before[split]

    def test_same_count_but_different_decks_is_reindexed(self, tmp_path: Path) -> None:
        box, uuids = _box_with_decks(5)
        _dealer(box, tmp_path)
        box.delete(uuids[0])
        replacement = _deck()
        box.create(replacement)

        dealer = _dealer(box, tmp_path)

        dealt = {uuid for split in Split for uuid in _dealt_uuids(dealer, split)}
        assert replacement.nocab_uuid in dealt
        assert uuids[0] not in dealt

    def test_an_index_from_before_deck_holdouts_is_rebuilt(
        self, tmp_path: Path
    ) -> None:
        box, _ = _box_with_decks(10)
        # An old-style index: an assignment with no index_info record
        index_path = tmp_path / "dealer.db"
        legacy = sqlite3.connect(str(index_path))
        legacy.execute(
            "CREATE TABLE deck_splits (deck_uuid TEXT PRIMARY KEY, split TEXT NOT NULL)"
        )
        legacy.execute("INSERT INTO deck_splits VALUES (?, ?)", (str(uuid4()), "test"))
        legacy.commit()
        legacy.close()

        dealer = _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 0, 0)))

        assert _split_counts(dealer) == (10, 0, 0)

    def test_a_crash_mid_rebuild_keeps_the_previous_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _assign_splits() runs the delete, every batch and the index_info
        # write in one transaction: a failure partway through a multi-batch
        # rebuild leaves the previous assignment, still recorded as built
        # from the previous spec - never a mix.
        import src.dojos.file_managers.deck_box_dealer as deck_box_dealer_module

        box, _ = _box_with_decks(5)
        _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 0, 0)))
        monkeypatch.setattr(deck_box_dealer_module, "_ASSIGN_SPLITS_BATCH_SIZE", 2)
        call_count = 0
        real_insert_split_batch = DeckBoxDealer._insert_split_batch

        def _raise_on_second_batch(self: DeckBoxDealer, batch: list) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated crash mid-rebuild")
            real_insert_split_batch(self, batch)

        monkeypatch.setattr(
            DeckBoxDealer, "_insert_split_batch", _raise_on_second_batch
        )

        with pytest.raises(RuntimeError, match="simulated crash"):
            _dealer(box, tmp_path, _holdout(deck_split_ratios=(0, 1, 0)))
        assert call_count == 2, "the fault must land mid-multi-batch"

        monkeypatch.undo()
        verification = sqlite3.connect(str(tmp_path / "dealer.db"))
        rows = verification.execute(
            "SELECT split, COUNT(*) FROM deck_splits GROUP BY split"
        ).fetchall()
        verification.close()
        assert dict(rows) == {"train": 5}

        # A retry under the new spec still sees the index as stale
        retried = _dealer(box, tmp_path, _holdout(deck_split_ratios=(0, 1, 0)))
        assert _split_counts(retried) == (0, 5, 0)


class TestPartition:
    def test_every_deck_lands_in_its_holdout_split(self, tmp_path: Path) -> None:
        box, uuids = _box_with_decks(50)
        holdout = _holdout(deck_split_ratios=(1, 1, 1))
        dealer = _dealer(box, tmp_path, holdout)

        for split in Split:
            expected = {
                uuid
                for uuid in uuids
                if holdout.deck_split_of(uuid, GameId.MTG) == split
            }
            assert set(_dealt_uuids(dealer, split)) == expected

    def test_every_deck_is_dealt_exactly_once_across_splits(
        self, tmp_path: Path
    ) -> None:
        box, uuids = _box_with_decks(30)
        dealer = _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 1, 1)))

        dealt = [uuid for split in Split for uuid in _dealt_uuids(dealer, split)]

        assert sorted(dealt) == sorted(uuids)

    def test_a_held_out_games_decks_are_all_validation(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(10)
        holdout = _holdout(held_out_games=frozenset({GameId.MTG}))

        dealer = _dealer(box, tmp_path, holdout)

        assert _split_counts(dealer) == (0, 0, 10)

    def test_deck_count_matches_decks_dealt(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(30)
        dealer = _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 1, 1)))

        for split in Split:
            assert dealer.deck_count(split) == len(_dealt_uuids(dealer, split))


class TestDecksForSplit:
    def test_raises_on_non_positive_decks_per_sample(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(3)
        dealer = _dealer(box, tmp_path)

        with pytest.raises(ValueError):
            list(dealer.training_decks(0))

    def test_drops_a_trailing_partial_group(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(10)
        dealer = _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 0, 0)))

        groups = list(dealer.training_decks(decks_per_sample=3))

        assert len(groups) == 3
        assert all(len(group) == 3 for group in groups)

    def test_the_fixed_order_is_by_deck_uuid(self, tmp_path: Path) -> None:
        box, uuids = _box_with_decks(12)
        dealer = _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 0, 0)))

        dealt = _dealt_uuids(dealer, Split.TRAIN)

        assert dealt == sorted(uuids, key=str)
        assert dealt == _dealt_uuids(dealer, Split.TRAIN)

    def test_shuffle_true_preserves_membership_across_reshuffles(
        self, tmp_path: Path
    ) -> None:
        box, _ = _box_with_decks(20)
        dealer = _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 0, 0)))

        first = {
            deck.nocab_uuid
            for group in dealer.training_decks(1, shuffle=True)
            for deck in group
        }
        second = {
            deck.nocab_uuid
            for group in dealer.training_decks(1, shuffle=True)
            for deck in group
        }

        assert first == second

    def test_groups_hold_each_decks_full_content(self, tmp_path: Path) -> None:
        box, uuids = _box_with_decks(6)
        dealer = _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 0, 0)))

        dealt = [deck for group in dealer.training_decks(3) for deck in group]

        assert sorted(dealt, key=lambda deck: str(deck.nocab_uuid)) == [
            box.get_by_uuid(nocab_uuid) for nocab_uuid in sorted(uuids, key=str)
        ]

    def test_a_deck_deleted_after_indexing_raises(self, tmp_path: Path) -> None:
        box, uuids = _box_with_decks(4)
        dealer = _dealer(box, tmp_path, _holdout(deck_split_ratios=(1, 0, 0)))
        box.delete(uuids[0])

        with pytest.raises(RuntimeError, match="no longer resolves"):
            list(dealer.training_decks(4))
