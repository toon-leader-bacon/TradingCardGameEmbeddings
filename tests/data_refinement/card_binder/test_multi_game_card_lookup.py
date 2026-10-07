"""Tests for multi_game_card_lookup.py's MultiGameCardLookup."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.multi_game_card_lookup import MultiGameCardLookup
from src.data_refinement.card_binder.visible_card_lookup import VisibleCardLookup
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split


def _card(game: GameId, name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=game,
        name=name,
        raw_content={"rarity": "common"},
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _binder(*cards: GenericCard) -> CardBinder:
    binder = CardBinder()
    for card in cards:
        binder.create(card)
    return binder


@pytest.fixture
def cards() -> tuple[GenericCard, GenericCard]:
    return _card(GameId.GWENT, "Ember"), _card(GameId.MTG, "Bolt")


@pytest.fixture
def lookup(cards: tuple[GenericCard, GenericCard]) -> MultiGameCardLookup:
    gwent, mtg = cards
    return MultiGameCardLookup({GameId.GWENT: _binder(gwent), GameId.MTG: _binder(mtg)})


class TestMultiGameCardLookup:
    def test_an_empty_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one game"):
            MultiGameCardLookup({})

    def test_get_by_uuid_finds_a_card_in_any_game(
        self, lookup: MultiGameCardLookup, cards: tuple[GenericCard, GenericCard]
    ) -> None:
        gwent, mtg = cards

        assert lookup.get_by_uuid(gwent.nocab_uuid) == gwent
        assert lookup.get_by_uuid(mtg.nocab_uuid) == mtg
        assert lookup.get_by_uuid(uuid4()) is None

    def test_game_keyed_calls_go_to_that_games_lookup(
        self, lookup: MultiGameCardLookup, cards: tuple[GenericCard, GenericCard]
    ) -> None:
        gwent, _ = cards

        assert lookup.get_by_name(GameId.GWENT, "Ember") == [gwent]
        assert lookup.get_by_name_single(GameId.GWENT, "Ember") == gwent
        assert lookup.get_by_name_regex(GameId.GWENT, "^Em") == [gwent]
        assert list(lookup.all_cards(GameId.GWENT)) == [gwent]
        assert lookup.get_by_name(GameId.MTG, "Ember") == []

    def test_a_game_it_does_not_hold_is_a_miss(
        self, lookup: MultiGameCardLookup
    ) -> None:
        assert lookup.get_by_name(GameId.DOMINION, "Village") == []
        assert lookup.get_by_name_single(GameId.DOMINION, "Village") is None
        assert lookup.get_by_alias(GameId.DOMINION, DataSource.GWENT_ONE, "x") is None
        assert lookup.get_by_name_regex(GameId.DOMINION, ".") == []
        assert list(lookup.all_cards(GameId.DOMINION)) == []
        assert list(lookup.all_uuids(GameId.DOMINION)) == []

    def test_all_uuids_without_a_game_chains_every_game(
        self, lookup: MultiGameCardLookup, cards: tuple[GenericCard, GenericCard]
    ) -> None:
        gwent, mtg = cards

        assert list(lookup.all_uuids()) == [gwent.nocab_uuid, mtg.nocab_uuid]
        assert list(lookup.all_uuids(GameId.MTG)) == [mtg.nocab_uuid]

    def test_version_for_matches_the_games_own_binder(self) -> None:
        gwent = _binder(_card(GameId.GWENT, "Ember"))
        lookup = MultiGameCardLookup({GameId.GWENT: gwent})

        assert lookup.version_for(GameId.GWENT) == gwent.version_for(GameId.GWENT)

    def test_version_for_an_unknown_game_raises(
        self, lookup: MultiGameCardLookup
    ) -> None:
        with pytest.raises(ValueError, match="dominion"):
            lookup.version_for(GameId.DOMINION)

    def test_works_under_a_visible_card_lookup(
        self, lookup: MultiGameCardLookup, cards: tuple[GenericCard, GenericCard]
    ) -> None:
        gwent, _ = cards
        view = VisibleCardLookup(lookup, HoldoutSpec.no_holdout(), Split.TRAIN)

        assert view.get_by_uuid(gwent.nocab_uuid) == gwent
