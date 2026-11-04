from uuid import UUID, uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.evaluation.intrinsic.deck_retrieval.evaluation_decks import (
    EvaluationDeck,
    EvaluationDeckSpec,
    sample_evaluation_decks,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec, UnknownDeckSplitError
from src.schema.splits import Split


def _holdout(
    deck_split_ratios: tuple[float, float, float] | None = (0, 0, 1),
    held_out_games: frozenset[GameId] = frozenset(),
) -> HoldoutSpec:
    return HoldoutSpec(
        seed=3,
        tier_ratios=(1, 0, 0),
        held_out_games=held_out_games,
        deck_split_ratios=deck_split_ratios,
    )


def _deck(cards: list[UUID], game: GameId = GameId.MTG) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(), source_game=game, name="deck", card_nocab_uuids=cards
    )


def _box(decks: list[GenericDeck]) -> DeckBox:
    box = DeckBox.load([])
    for deck in decks:
        box.create(deck)
    return box


def _decks_of_size(count: int, size: int = 4) -> list[GenericDeck]:
    return [_deck([uuid4() for _ in range(size)]) for _ in range(count)]


def _spec(count: int = 100, seed: int = 0, min_cards: int = 4) -> EvaluationDeckSpec:
    return EvaluationDeckSpec(count=count, seed=seed, min_cards=min_cards)


class TestSpec:
    def test_rejects_count_below_one(self) -> None:
        with pytest.raises(ValueError, match="count"):
            EvaluationDeckSpec(count=0, seed=0)

    def test_rejects_min_cards_below_two(self) -> None:
        with pytest.raises(ValueError, match="min_cards"):
            EvaluationDeckSpec(count=1, seed=0, min_cards=1)


class TestSampleEvaluationDecks:
    def test_docstring_example_returns_the_requested_count(self) -> None:
        box = _box(_decks_of_size(30))
        decks = sample_evaluation_decks(box, GameId.MTG, _holdout(), _spec(count=10))
        assert len(decks) == 10
        assert all(isinstance(deck, EvaluationDeck) for deck in decks)

    def test_returns_only_validation_decks(self) -> None:
        holdout = _holdout(deck_split_ratios=(1, 1, 1))
        box = _box(_decks_of_size(60))
        decks = sample_evaluation_decks(box, GameId.MTG, holdout, _spec())
        assert decks
        assert all(
            holdout.deck_split_of(deck.deck_uuid, GameId.MTG) == Split.VALIDATION
            for deck in decks
        )
        validation_count = sum(
            holdout.deck_split_of(uuid, GameId.MTG) == Split.VALIDATION
            for uuid in box.all_uuids(GameId.MTG)
        )
        assert len(decks) == validation_count

    def test_no_validation_decks_gives_empty(self) -> None:
        box = _box(_decks_of_size(10))
        holdout = _holdout(deck_split_ratios=(1, 0, 0))
        assert sample_evaluation_decks(box, GameId.MTG, holdout, _spec()) == ()

    def test_held_out_game_decks_are_all_validation(self) -> None:
        box = _box(_decks_of_size(10))
        holdout = _holdout(
            deck_split_ratios=(1, 0, 0), held_out_games=frozenset({GameId.MTG})
        )
        assert len(sample_evaluation_decks(box, GameId.MTG, holdout, _spec())) == 10

    def test_only_the_given_game(self) -> None:
        box = _box(
            _decks_of_size(5) + [_deck([uuid4() for _ in range(4)], GameId.GWENT)]
        )
        decks = sample_evaluation_decks(box, GameId.GWENT, _holdout(), _spec())
        assert [deck.source_game for deck in decks] == [GameId.GWENT]

    def test_same_seed_same_decks_regardless_of_insertion_order(self) -> None:
        decks = _decks_of_size(40)
        forward = sample_evaluation_decks(
            _box(decks), GameId.MTG, _holdout(), _spec(10)
        )
        backward = sample_evaluation_decks(
            _box(list(reversed(decks))), GameId.MTG, _holdout(), _spec(10)
        )
        assert forward == backward

    def test_different_seed_changes_the_order(self) -> None:
        box = _box(_decks_of_size(40))
        first = sample_evaluation_decks(box, GameId.MTG, _holdout(), _spec(seed=0))
        second = sample_evaluation_decks(box, GameId.MTG, _holdout(), _spec(seed=1))
        assert [d.deck_uuid for d in first] != [d.deck_uuid for d in second]
        assert {d.deck_uuid for d in first} == {d.deck_uuid for d in second}

    def test_cards_are_distinct_in_deck_order_without_the_placeholder(self) -> None:
        a, b, c, d = (uuid4() for _ in range(4))
        placeholder = CardBinder.unknown_card_uuid(GameId.MTG)
        deck = _deck([b, a, b, placeholder, c, a, d])
        (sampled,) = sample_evaluation_decks(
            _box([deck]), GameId.MTG, _holdout(), _spec()
        )
        assert sampled == EvaluationDeck(deck.nocab_uuid, GameId.MTG, (b, a, c, d))

    def test_min_cards_counts_distinct_known_cards(self) -> None:
        a, b, c = (uuid4() for _ in range(3))
        placeholder = CardBinder.unknown_card_uuid(GameId.MTG)
        too_small = _deck([a, a, b, c, placeholder])
        box = _box([too_small])
        assert sample_evaluation_decks(box, GameId.MTG, _holdout(), _spec()) == ()
        assert (
            len(
                sample_evaluation_decks(box, GameId.MTG, _holdout(), _spec(min_cards=3))
            )
            == 1
        )

    def test_small_decks_do_not_count_toward_count(self) -> None:
        box = _box(_decks_of_size(20, size=2) + _decks_of_size(5, size=4))
        decks = sample_evaluation_decks(box, GameId.MTG, _holdout(), _spec(count=5))
        assert len(decks) == 5

    def test_shortfall_logs_and_returns_fewer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        box = _box(_decks_of_size(3))
        with caplog.at_level("INFO"):
            decks = sample_evaluation_decks(
                box, GameId.MTG, _holdout(), _spec(count=10)
            )
        assert len(decks) == 3
        assert "3 of 10" in caplog.text

    def test_unknown_deck_split_raises(self) -> None:
        box = _box(_decks_of_size(3))
        with pytest.raises(UnknownDeckSplitError):
            sample_evaluation_decks(
                box, GameId.MTG, _holdout(deck_split_ratios=None), _spec()
            )

    def test_unknown_deck_split_raises_even_with_no_decks(self) -> None:
        with pytest.raises(UnknownDeckSplitError):
            sample_evaluation_decks(
                _box([]), GameId.MTG, _holdout(deck_split_ratios=None), _spec()
            )
