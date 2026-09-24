from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.contrastive.deck_box_dealer import DeckBoxDealer
from src.schema.card import GenericDeck
from src.schema.game_id import GameId


def _deck(card_count: int = 1) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="deck",
        card_nocab_uuids=[uuid4() for _ in range(card_count)],
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


class TestSourceGameAndCardBinderVersion:
    def test_source_game_is_the_constructor_argument(self) -> None:
        box, _ = _box_with_decks(1)

        dealer = DeckBoxDealer(box, GameId.MTG)

        assert dealer.source_game == GameId.MTG

    def test_card_binder_version_is_none_for_an_unstamped_box(self) -> None:
        box, _ = _box_with_decks(1)

        dealer = DeckBoxDealer(box, GameId.MTG)

        assert dealer.card_binder_version is None

    def test_card_binder_version_reflects_the_box(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(1)
        path = tmp_path / "mtg.jsonl"
        box.save(path, GameId.MTG, "binder-v1")
        loaded = DeckBox.load([path])

        dealer = DeckBoxDealer(loaded, GameId.MTG)

        assert dealer.card_binder_version == "binder-v1"


class TestInit:
    def test_raises_on_wrong_split_ratios_length(self) -> None:
        box, _ = _box_with_decks(1)

        with pytest.raises(ValueError):
            DeckBoxDealer(box, GameId.MTG, split_ratios=[1, 1])

    def test_raises_on_negative_split_ratio(self) -> None:
        box, _ = _box_with_decks(1)

        with pytest.raises(ValueError):
            DeckBoxDealer(box, GameId.MTG, split_ratios=[8, -1, 1])

    def test_raises_on_all_zero_split_ratios(self) -> None:
        box, _ = _box_with_decks(1)

        with pytest.raises(ValueError):
            DeckBoxDealer(box, GameId.MTG, split_ratios=[0, 0, 0])

    def test_only_considers_the_given_game(self) -> None:
        box = DeckBox()
        mtg_deck = _deck()
        box.create(mtg_deck)
        other_deck = GenericDeck(
            nocab_uuid=uuid4(),
            source_game=GameId.POKEMON,
            name="other",
            card_nocab_uuids=[uuid4()],
            provenance=None,
        )
        box.create(other_deck)

        dealer = DeckBoxDealer(box, GameId.MTG, seed=1)

        all_dealt = (
            list(dealer.training_decks(1))
            + list(dealer.test_decks(1))
            + list(dealer.validation_decks(1))
        )
        dealt_uuids = {deck.nocab_uuid for group in all_dealt for deck in group}
        assert dealt_uuids == {mtg_deck.nocab_uuid}


class TestPartition:
    def test_splits_proportionally_to_ratios(self) -> None:
        box, uuids = _box_with_decks(10)
        dealer = DeckBoxDealer(box, GameId.MTG, split_ratios=[8, 1, 1], seed=1)

        train_count = len(list(dealer.training_decks(1)))
        test_count = len(list(dealer.test_decks(1)))
        validation_count = len(list(dealer.validation_decks(1)))

        assert (train_count, test_count, validation_count) == (8, 1, 1)

    def test_every_deck_is_dealt_exactly_once_across_splits(self) -> None:
        box, uuids = _box_with_decks(10)
        dealer = DeckBoxDealer(box, GameId.MTG, split_ratios=[8, 1, 1], seed=1)

        dealt = (
            [deck for group in dealer.training_decks(1) for deck in group]
            + [deck for group in dealer.test_decks(1) for deck in group]
            + [deck for group in dealer.validation_decks(1) for deck in group]
        )

        assert sorted(deck.nocab_uuid for deck in dealt) == sorted(uuids)


class TestDecksForSplit:
    def test_raises_on_non_positive_decks_per_sample(self) -> None:
        box, _ = _box_with_decks(3)
        dealer = DeckBoxDealer(box, GameId.MTG, seed=1)

        with pytest.raises(ValueError):
            list(dealer.training_decks(0))

    def test_drops_a_trailing_partial_group(self) -> None:
        box, _ = _box_with_decks(10)
        # All 10 decks land in training with these ratios.
        dealer = DeckBoxDealer(box, GameId.MTG, split_ratios=[1, 0, 0], seed=1)

        groups = list(dealer.training_decks(decks_per_sample=3))

        assert len(groups) == 3
        assert all(len(group) == 3 for group in groups)

    def test_is_deterministic_given_the_same_seed(self) -> None:
        box, _ = _box_with_decks(9)
        dealer_a = DeckBoxDealer(box, GameId.MTG, split_ratios=[1, 0, 0], seed=7)
        dealer_b = DeckBoxDealer(box, GameId.MTG, split_ratios=[1, 0, 0], seed=7)

        groups_a = [
            [deck.nocab_uuid for deck in group]
            for group in dealer_a.training_decks(3, shuffle=True)
        ]
        groups_b = [
            [deck.nocab_uuid for deck in group]
            for group in dealer_b.training_decks(3, shuffle=True)
        ]

        assert groups_a == groups_b
