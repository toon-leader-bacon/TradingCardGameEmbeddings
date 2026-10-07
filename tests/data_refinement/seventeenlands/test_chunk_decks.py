"""Tests for chunk_decks.py's build_chunk_decks(): each distinct deck is
identified once, with the row implementation's deck id and name, and a
row's deck is the full copy-count multiset (not merely which columns
are present)."""

from uuid import uuid4

import numpy as np

from src.data_refinement.deck_ids import deck_uuid_from_cards
from src.data_refinement.seventeenlands.chunk_decks import (
    GameKeys,
    build_chunk_decks,
)
from src.data_refinement.seventeenlands.zone_counts import ZoneCounts
from src.schema.game_id import GameId


def _keys(rows: int) -> GameKeys:
    return GameKeys(
        draft_id=np.array([f"d{i}" for i in range(rows)], object),
        match_number=np.arange(rows, dtype=np.int64),
        game_number=np.full(rows, 2, np.int64),
    )


def test_rows_with_the_same_copy_counts_share_one_deck() -> None:
    owlbear, morningstar = uuid4(), uuid4()
    zone = ZoneCounts(
        (owlbear, morningstar), np.array([[4, 1], [0, 2], [4, 1]], np.int16)
    )

    decks = build_chunk_decks(zone, _keys(3), GameId.MTG, "game_data")

    assert len(decks.decks) == 2
    assert decks.row_deck.tolist() == [0, 1, 0]
    assert decks.decks[0].card_nocab_uuids == [owlbear] * 4 + [morningstar]
    assert decks.decks[1].card_nocab_uuids == [morningstar, morningstar]


def test_rows_with_the_same_present_columns_but_different_counts_differ() -> None:
    # Regression test: presence alone (count > 0) used to be the
    # pre-grouping key, which would have wrongly merged these two rows
    # before hashing even though their copy counts - and therefore
    # their multisets and deck ids - differ.
    owlbear, morningstar = uuid4(), uuid4()
    zone = ZoneCounts((owlbear, morningstar), np.array([[4, 1], [1, 3]], np.int16))

    decks = build_chunk_decks(zone, _keys(2), GameId.MTG, "game_data")

    assert len(decks.decks) == 2
    assert decks.row_deck.tolist() == [0, 1]
    assert decks.decks[0].nocab_uuid != decks.decks[1].nocab_uuid
    assert decks.decks[0].card_nocab_uuids == [owlbear] * 4 + [morningstar]
    assert decks.decks[1].card_nocab_uuids == [owlbear] + [morningstar] * 3


def test_deck_ids_are_the_row_implementations_hash() -> None:
    owlbear, morningstar = uuid4(), uuid4()
    zone = ZoneCounts((owlbear, morningstar), np.array([[1, 1]], np.int16))

    (deck,) = build_chunk_decks(zone, _keys(1), GameId.MTG, "game_data").decks

    assert deck.nocab_uuid == deck_uuid_from_cards([owlbear, morningstar])
    assert deck.source_game is GameId.MTG


def test_a_card_in_two_columns_is_listed_twice() -> None:
    owlbear = uuid4()
    zone = ZoneCounts((owlbear, owlbear), np.array([[1, 1]], np.int16))

    (deck,) = build_chunk_decks(zone, _keys(1), GameId.MTG, "game_data").decks

    assert deck.card_nocab_uuids == [owlbear, owlbear]
    assert deck.nocab_uuid == deck_uuid_from_cards([owlbear, owlbear])


def test_each_deck_is_named_after_its_first_row() -> None:
    owlbear = uuid4()
    zone = ZoneCounts((owlbear,), np.array([[0], [1], [1]], np.int16))

    decks = build_chunk_decks(zone, _keys(3), GameId.MTG, "game_data")

    assert [deck.name for deck in decks.decks] == [
        "game_data d0/0/2 deck",
        "game_data d1/1/2 deck",
    ]


def test_an_empty_deck_zone_gives_every_row_the_empty_deck() -> None:
    zone = ZoneCounts((), np.zeros((2, 0), np.int16))

    decks = build_chunk_decks(zone, _keys(2), GameId.MTG, "game_data")

    assert decks.row_deck.tolist() == [0, 0]
    assert decks.decks[0].nocab_uuid == deck_uuid_from_cards([])


def test_no_rows_gives_no_decks() -> None:
    zone = ZoneCounts((uuid4(),), np.zeros((0, 1), np.int16))

    decks = build_chunk_decks(zone, _keys(0), GameId.MTG, "game_data")

    assert decks.decks == ()
    assert decks.row_deck.shape == (0,)


def test_many_columns_group_correctly() -> None:
    # More than 8 columns, and row 2 repeats row 0's present columns at
    # a different copy count, so it must land in its own group/deck.
    cards = tuple(uuid4() for _ in range(20))
    counts = np.zeros((3, 20), np.int16)
    counts[0, [0, 9, 19]] = 1
    counts[1, [0, 9]] = 1
    counts[2, [0, 9, 19]] = 2

    decks = build_chunk_decks(
        ZoneCounts(cards, counts), _keys(3), GameId.MTG, "game_data"
    )

    assert decks.row_deck.tolist() == [0, 1, 2]
    assert decks.decks[0].card_nocab_uuids == [cards[0], cards[9], cards[19]]
    assert decks.decks[2].card_nocab_uuids == [
        cards[0],
        cards[0],
        cards[9],
        cards[9],
        cards[19],
        cards[19],
    ]


def test_patterns_hashing_to_one_deck_share_its_earliest_entry() -> None:
    # Two columns name one card: rows 0 and 1 differ in pattern but play
    # the same one-card deck, so they share row 0's deck
    owlbear = uuid4()
    zone = ZoneCounts((owlbear, owlbear), np.array([[1, 0], [0, 1]], np.int16))

    decks = build_chunk_decks(zone, _keys(2), GameId.MTG, "game_data")

    assert decks.row_deck.tolist() == [0, 0]
    assert len(decks.decks) == 1
    assert decks.decks[0].name == "game_data d0/0/2 deck"
