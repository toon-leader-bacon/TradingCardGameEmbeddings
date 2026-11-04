import threading
from collections import Counter
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.file_managers.deck_box_dealer import DeckBoxDealer
from src.dojos.file_managers.deck_split_index import DeckSplitIndex
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from tests.snapshot_fixtures import (
    binder_with_cards,
    built_snapshots,
    dealer_over,
    stamped_deck_box,
)


def _decks(count: int, binder: CardBinder | None = None) -> list[GenericDeck]:
    card_uuids = list(binder.all_uuids(GameId.MTG)) if binder else [uuid4()]
    return [
        GenericDeck(
            nocab_uuid=uuid4(),
            source_game=GameId.MTG,
            name="deck",
            card_nocab_uuids=card_uuids,
            provenance=None,
        )
        for _ in range(count)
    ]


def _holdout(
    deck_split_ratios: tuple[float, float, float] | None = (8, 1, 1),
    held_out_games: frozenset[GameId] = frozenset(),
) -> HoldoutSpec:
    return HoldoutSpec(
        seed=1,
        tier_ratios=(1, 0, 0),
        held_out_games=held_out_games,
        deck_split_ratios=deck_split_ratios,
    )


def _dealt_uuids(
    dealer: DeckBoxDealer, split: Split, shuffle: bool = False
) -> list[UUID]:
    return [
        deck.nocab_uuid
        for group in dealer.decks_for(split, 1, shuffle=shuffle)
        for deck in group
    ]


def _split_counts(dealer: DeckBoxDealer) -> tuple[int, int, int]:
    return (
        dealer.deck_count(Split.TRAIN),
        dealer.deck_count(Split.TEST),
        dealer.deck_count(Split.VALIDATION),
    )


class TestProperties:
    def test_source_game_is_the_snapshots(self) -> None:
        assert dealer_over(_decks(1), _holdout()).source_game == GameId.MTG

    def test_card_binder_version_is_the_card_snapshots(self) -> None:
        binder = binder_with_cards(2)

        dealer = dealer_over(_decks(1, binder), _holdout(), binder)

        assert dealer.card_binder_version == binder.version_for(GameId.MTG)

    def test_deck_split_key_is_the_holdouts_for_this_game(self) -> None:
        holdout = _holdout()
        dealer = dealer_over(_decks(1), holdout)
        assert dealer.deck_split_key == holdout.deck_split_key(GameId.MTG)


class TestInit:
    def test_a_split_index_of_another_snapshot_is_refused(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        first = built_snapshots(
            binder, stamped_deck_box(_decks(2), binder, tmp_path / "a"), tmp_path / "a"
        )
        second = built_snapshots(
            binder, stamped_deck_box(_decks(3), binder, tmp_path / "b"), tmp_path / "b"
        )
        index = DeckSplitIndex.load_or_build(first, _holdout(), tmp_path / "splits")

        with pytest.raises(ValueError, match="another"):
            DeckBoxDealer(second, index, shuffle_seed=0)


class TestPartition:
    def test_every_deck_lands_in_its_holdout_split(self) -> None:
        decks = _decks(50)
        holdout = _holdout(deck_split_ratios=(1, 1, 1))
        dealer = dealer_over(decks, holdout)

        for split in Split:
            expected = {
                deck.nocab_uuid
                for deck in decks
                if holdout.deck_split_of(deck.nocab_uuid, GameId.MTG) == split
            }
            assert set(_dealt_uuids(dealer, split)) == expected

    def test_every_deck_is_dealt_exactly_once_across_splits(self) -> None:
        decks = _decks(30)
        dealer = dealer_over(decks, _holdout(deck_split_ratios=(1, 1, 1)))

        dealt = [uuid for split in Split for uuid in _dealt_uuids(dealer, split)]

        assert sorted(dealt) == sorted(deck.nocab_uuid for deck in decks)

    def test_a_held_out_games_decks_are_all_validation(self) -> None:
        holdout = _holdout(held_out_games=frozenset({GameId.MTG}))
        assert _split_counts(dealer_over(_decks(10), holdout)) == (0, 0, 10)

    def test_deck_count_matches_decks_dealt(self) -> None:
        dealer = dealer_over(_decks(30), _holdout(deck_split_ratios=(1, 1, 1)))
        for split in Split:
            assert dealer.deck_count(split) == len(_dealt_uuids(dealer, split))


class TestDecksForSplit:
    def test_raises_on_non_positive_decks_per_sample(self) -> None:
        dealer = dealer_over(_decks(3), _holdout())
        with pytest.raises(ValueError):
            list(dealer.training_decks(0))

    def test_drops_a_trailing_partial_group(self) -> None:
        dealer = dealer_over(_decks(10), _holdout(deck_split_ratios=(1, 0, 0)))

        groups = list(dealer.training_decks(decks_per_sample=3))

        assert [len(group) for group in groups] == [3, 3, 3]

    def test_keep_remainder_yields_a_shorter_last_group(self) -> None:
        dealer = dealer_over(_decks(10), _holdout(deck_split_ratios=(1, 0, 0)))

        groups = list(dealer.decks_for(Split.TRAIN, 3, keep_remainder=True))

        assert [len(group) for group in groups] == [3, 3, 3, 1]

    def test_an_empty_split_deals_nothing(self) -> None:
        dealer = dealer_over(_decks(4), _holdout(deck_split_ratios=(1, 0, 0)))
        assert list(dealer.test_decks(2)) == []
        assert list(dealer.validation_decks(2, shuffle=False)) == []

    def test_the_fixed_order_is_by_deck_uuid(self) -> None:
        decks = _decks(12)
        dealer = dealer_over(decks, _holdout(deck_split_ratios=(1, 0, 0)))

        dealt = _dealt_uuids(dealer, Split.TRAIN)

        assert dealt == sorted((deck.nocab_uuid for deck in decks), key=str)
        assert dealt == _dealt_uuids(dealer, Split.TRAIN)

    def test_shuffles_keep_membership_and_change_order(self) -> None:
        dealer = dealer_over(_decks(40), _holdout(deck_split_ratios=(1, 0, 0)))

        first = _dealt_uuids(dealer, Split.TRAIN, shuffle=True)
        second = _dealt_uuids(dealer, Split.TRAIN, shuffle=True)

        assert set(first) == set(second)
        assert first != second

    def test_the_same_seed_gives_the_same_epochs(self, tmp_path: Path) -> None:
        decks = _decks(40)
        holdout = _holdout(deck_split_ratios=(1, 0, 0))
        dealer_a = dealer_over(decks, holdout, root=tmp_path / "a", shuffle_seed=7)
        dealer_b = dealer_over(decks, holdout, root=tmp_path / "b", shuffle_seed=7)

        for _ in range(2):
            assert _dealt_uuids(dealer_a, Split.TRAIN, shuffle=True) == _dealt_uuids(
                dealer_b, Split.TRAIN, shuffle=True
            )

    def test_dealt_decks_hold_each_decks_card_multiset(self) -> None:
        binder = binder_with_cards(4)
        card_uuids = list(binder.all_uuids(GameId.MTG))
        decks = [
            GenericDeck(uuid4(), GameId.MTG, "named", card_uuids[:2] * 3),
            GenericDeck(uuid4(), GameId.MTG, "named", card_uuids),
        ]
        dealer = dealer_over(decks, _holdout(deck_split_ratios=(1, 0, 0)), binder)

        dealt = {
            deck.nocab_uuid: deck
            for group in dealer.training_decks(1, shuffle=False)
            for deck in group
        }

        for deck in decks:
            rebuilt = dealt[deck.nocab_uuid]
            assert Counter(rebuilt.card_nocab_uuids) == Counter(deck.card_nocab_uuids)
            assert rebuilt.source_game == GameId.MTG
            assert rebuilt.name == ""
            assert rebuilt.provenance is None


class TestReadFromAnotherThread:
    def test_a_dealer_built_on_one_thread_deals_on_another(self) -> None:
        # The prefetch worker's case: built here, read on a worker thread
        decks = _decks(5)
        dealer = dealer_over(decks, _holdout(deck_split_ratios=(1, 0, 0)))
        dealt: list[UUID] = []
        counted: list[int] = []

        def deal() -> None:
            counted.append(dealer.deck_count(Split.TRAIN))
            dealt.extend(_dealt_uuids(dealer, Split.TRAIN, shuffle=True))

        worker = threading.Thread(target=deal)
        worker.start()
        worker.join(timeout=5)

        assert counted == [5]
        assert sorted(dealt) == sorted(deck.nocab_uuid for deck in decks)
