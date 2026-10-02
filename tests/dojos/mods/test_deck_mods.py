import copy
import logging
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.dojos.mods.deck_mods import (
    CardDropoutMod,
    CardSubsampleMod,
    DeckThinningMod,
    DuplicateCollapseMod,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.DOMINION,
        name=name,
        raw_content={"name": name},
        provenance=Provenance(
            DataSource.DOMINIONTABS, name, datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )


def _deck(size: int) -> list[GenericCard]:
    return [_card(f"card {index}") for index in range(size)]


_EVERY_MOD = [
    lambda groups: CardDropoutMod(0.5, groups, rng_seed=0),
    lambda groups: CardSubsampleMod(0.5, groups, rng_seed=0),
    lambda groups: DuplicateCollapseMod(groups),
]


def _is_subsequence(kept: list[GenericCard], deck: list[GenericCard]) -> bool:
    """Whether kept is deck with some cards removed, order unchanged."""
    remaining = iter(deck)
    return all(any(card is other for other in remaining) for card in kept)


class TestNonMutation:
    @pytest.mark.parametrize("make_mod", _EVERY_MOD)
    def test_a_multi_card_datum_is_left_unchanged(self, make_mod) -> None:
        copper = _card("Copper")
        deck = [copper, *_deck(10), copper]
        before = list(deck)
        snapshot = copy.deepcopy(deck)
        datum = (deck, 1.0)

        for _ in range(20):
            make_mod(frozenset({0})).apply_single(datum)

        assert datum[0] is deck
        assert all(card is old for card, old in zip(deck, before))
        assert deck == snapshot

    @pytest.mark.parametrize("make_mod", _EVERY_MOD)
    def test_a_multi_group_datum_is_left_unchanged(self, make_mod) -> None:
        copper = _card("Copper")
        groups = [[copper, copper, *_deck(5)], [copper, *_deck(6)]]
        snapshot = [list(group) for group in groups]

        result, _ = make_mod(frozenset({0, 1})).apply_single((groups, 0.0))

        assert result is not groups
        assert [list(group) for group in groups] == snapshot
        assert all(len(group) == len(old) for group, old in zip(groups, snapshot))

    def test_nothing_removed_returns_the_same_datum(self) -> None:
        datum = (_deck(4), 2.0)
        assert CardDropoutMod(0.0, rng_seed=0).apply_single(datum) is datum
        assert CardSubsampleMod(1.0, rng_seed=0).apply_single(datum) is datum
        assert DuplicateCollapseMod().apply_single(datum) is datum

    def test_kept_cards_keep_their_order_and_the_label(self) -> None:
        deck = _deck(20)
        kept, label = CardDropoutMod(0.5, rng_seed=1).apply_single((deck, 3))
        assert label == 3
        assert 0 < len(kept) < len(deck)
        assert _is_subsequence(kept, deck)


class TestNeverEmpty:
    def test_full_dropout_keeps_one_card(self) -> None:
        mod = CardDropoutMod(1.0, rng_seed=0)
        for size in (1, 2, 30):
            deck = _deck(size)
            kept, _ = mod.apply_single((deck, 0))
            assert len(kept) == 1 and kept[0] in deck

    def test_a_tiny_fraction_keeps_one_card(self) -> None:
        kept, _ = CardSubsampleMod(0.01, rng_seed=0).apply_single((_deck(10), 0))
        assert len(kept) == 1

    def test_subsample_keeps_the_rounded_fraction(self) -> None:
        kept, _ = CardSubsampleMod(0.25, rng_seed=0).apply_single((_deck(40), 0))
        assert len(kept) == 10

    def test_every_thinned_group_keeps_a_card(self) -> None:
        mod = CardDropoutMod(1.0, frozenset({0, 1}), rng_seed=0)
        groups, _ = mod.apply_single(([_deck(5), _deck(3)], 0))
        assert [len(group) for group in groups] == [1, 1]

    def test_an_empty_group_passes_through(self) -> None:
        empty: list[GenericCard] = []
        datum = ([empty, _deck(3)], 0)
        groups, _ = CardDropoutMod(1.0, frozenset({0, 1}), rng_seed=0).apply_single(
            datum
        )
        assert groups[0] is empty and len(groups[1]) == 1


class TestGroups:
    def test_an_unlisted_group_is_the_same_list(self) -> None:
        options, context = _deck(6), _deck(12)
        mod = CardDropoutMod(1.0, frozenset({1}), rng_seed=0)
        for _ in range(10):
            (new_options, new_context), _ = mod.apply_single(([options, context], 2))
            assert new_options is options  # the label indexes this group
            assert len(new_context) == 1

    def test_a_multi_card_input_stays_a_flat_list(self) -> None:
        kept, _ = CardDropoutMod(1.0, rng_seed=0).apply_single((_deck(5), 0))
        assert isinstance(kept, list) and isinstance(kept[0], GenericCard)

    def test_duplicates_collapse_to_their_first_copy(self) -> None:
        copper, estate = _card("Copper"), _card("Estate")
        twin = GenericCard(**{**vars(copper)})  # same nocab_uuid, other object
        kept, _ = DuplicateCollapseMod().apply_single(([copper, estate, twin], 0))
        assert kept == [copper, estate] and kept[0] is copper


class TestBestEffort:
    @pytest.mark.parametrize(
        "training_input",
        [_card("Lone"), [_card("A"), [_card("B")]], [[_card("A")], ["not a card"]]],
    )
    def test_an_odd_input_passes_through_and_is_counted(
        self, training_input, caplog
    ) -> None:
        mod = CardDropoutMod(0.5, rng_seed=0)
        datum = (training_input, 1)
        with caplog.at_level(logging.WARNING):
            assert mod.apply_single(datum) is datum
        assert mod.tally.cards_failed == 1 and mod.tally.first_failure is not None
        assert "passed a datum through" in caplog.text

    def test_a_group_past_the_datum_is_a_failure(self) -> None:
        mod = CardDropoutMod(0.5, frozenset({1}), rng_seed=0)
        datum = (_deck(4), 0)  # a multi-card input has only group 0
        assert mod.apply_single(datum) is datum
        assert mod.tally.cards_failed == 1


class TestTally:
    def test_counts_cards_seen_and_removed(self) -> None:
        mod = CardDropoutMod(1.0, frozenset({1}), rng_seed=0)
        mod.apply_single(([_deck(3), _deck(10)], 0))
        mod.apply_single(([_deck(3), _deck(4)], 0))
        # group 0 is never counted; each group 1 keeps one card
        assert (mod.tally.cards_seen, mod.tally.cards_changed) == (14, 12)
        assert mod.tally.cards_failed == 0

    def test_dropout_rate_matches_its_probability(self) -> None:
        mod = CardDropoutMod(0.2, rng_seed=3)
        for _ in range(200):
            mod.apply_single((_deck(50), 0))
        assert mod.tally.changed_fraction == pytest.approx(0.2, abs=0.02)

    def test_collapse_counts_removed_copies(self) -> None:
        copper = _card("Copper")
        mod = DuplicateCollapseMod()
        mod.apply_single(([copper, copper, copper, _card("Estate")], 0))
        assert (mod.tally.cards_seen, mod.tally.cards_changed) == (4, 2)


class TestValidation:
    @pytest.mark.parametrize("probability", [-0.1, 1.5, float("nan")])
    def test_dropout_probability(self, probability: float) -> None:
        with pytest.raises(ValueError):
            CardDropoutMod(probability)

    @pytest.mark.parametrize("fraction", [0.0, 1.5, float("nan")])
    def test_keep_fraction(self, fraction: float) -> None:
        with pytest.raises(ValueError):
            CardSubsampleMod(fraction)

    @pytest.mark.parametrize(
        "groups", [frozenset(), frozenset({-1}), frozenset({True})]
    )
    def test_groups(self, groups: frozenset) -> None:
        with pytest.raises(ValueError):
            DuplicateCollapseMod(groups)

    def test_deck_mods_are_train_only(self) -> None:
        mod: DeckThinningMod = CardSubsampleMod(0.5)
        assert mod.train_only
