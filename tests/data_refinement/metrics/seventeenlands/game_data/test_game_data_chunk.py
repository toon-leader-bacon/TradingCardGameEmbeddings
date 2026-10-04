"""Tests for game_data_chunk.py: ZoneCounts, ChunkDecks and
GameDataChunk's invariants."""

from uuid import uuid4

import numpy as np
import pytest

from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_refinement.metrics.seventeenlands.chunk_decks import (
    ChunkDecks,
    GameKeys,
)
from src.data_refinement.metrics.seventeenlands.zone_counts import ZoneCounts
from src.schema.card import GenericDeck
from src.schema.game_id import GameId


def _empty_zone(rows: int) -> ZoneCounts:
    return ZoneCounts(card_uuids=(), counts=np.zeros((rows, 0), np.int16))


def _keys(rows: int) -> GameKeys:
    return GameKeys(
        draft_id=np.full(rows, "d", object),
        match_number=np.ones(rows, np.int64),
        game_number=np.ones(rows, np.int64),
    )


def _deck(name: str = "deck") -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(), source_game=GameId.MTG, name=name, card_nocab_uuids=[]
    )


def _chunk(
    rows: int = 2,
    zones: dict | None = None,
    num_turns_rows: int | None = None,
    rank_rows: int | None = None,
    key_rows: int | None = None,
) -> GameDataChunk:
    return GameDataChunk(
        zones=(
            zones
            if zones is not None
            else {zone: _empty_zone(rows) for zone in GameZone}
        ),
        won=np.zeros(rows, np.bool_),
        on_play=np.zeros(rows, np.bool_),
        num_turns=np.zeros(
            num_turns_rows if num_turns_rows is not None else rows, np.int32
        ),
        keys=_keys(key_rows if key_rows is not None else rows),
        rank=np.full(rank_rows if rank_rows is not None else rows, "", object),
        decks=ChunkDecks(decks=(_deck(),), row_deck=np.zeros(rows, np.intp)),
    )


def test_present_marks_cells_with_at_least_one_copy() -> None:
    zone = ZoneCounts(
        card_uuids=(uuid4(), uuid4()), counts=np.array([[0, 2], [1, 0]], np.int16)
    )

    assert zone.present().tolist() == [[False, True], [True, False]]


def test_present_for_lines_cards_up_across_their_columns() -> None:
    owlbear, morningstar, absent = uuid4(), uuid4(), uuid4()
    # Owlbear has two columns; a row holds it if either is present
    zone = ZoneCounts(
        card_uuids=(owlbear, morningstar, owlbear),
        counts=np.array([[1, 0, 0], [0, 0, 2], [0, 3, 0]], np.int16),
    )

    present = zone.present_for((morningstar, absent, owlbear, owlbear))

    assert present.tolist() == [
        [False, False, True, True],
        [False, False, True, True],
        [True, False, False, False],
    ]


def test_present_for_an_empty_zone_is_all_false() -> None:
    assert _empty_zone(2).present_for((uuid4(),)).tolist() == [[False], [False]]


def test_row_deck_uuids_maps_each_row_to_its_deck() -> None:
    first, second = _deck("a"), _deck("b")
    decks = ChunkDecks(decks=(first, second), row_deck=np.array([1, 0, 1], np.intp))

    assert decks.row_deck_uuids().tolist() == [
        str(second.nocab_uuid),
        str(first.nocab_uuid),
        str(second.nocab_uuid),
    ]


def test_two_decks_with_one_id_are_rejected() -> None:
    deck = _deck()

    with pytest.raises(ValueError, match="same id"):
        ChunkDecks(decks=(deck, deck), row_deck=np.zeros(1, np.intp))


def test_a_row_deck_outside_the_decks_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside its 1 decks"):
        ChunkDecks(decks=(_deck(),), row_deck=np.array([0, 1], np.intp))


def test_len_is_the_row_count() -> None:
    assert len(_chunk(rows=3)) == 3


def test_a_missing_zone_is_rejected() -> None:
    zones = {zone: _empty_zone(2) for zone in GameZone if zone is not GameZone.TUTORED}

    with pytest.raises(ValueError, match="TUTORED"):
        _chunk(zones=zones)


@pytest.mark.parametrize(
    "field, overrides",
    [
        ("num_turns", {"num_turns_rows": 3}),
        ("rank", {"rank_rows": 3}),
        ("keys.draft_id", {"key_rows": 3}),
    ],
)
def test_fields_with_different_row_counts_are_rejected(
    field: str, overrides: dict
) -> None:
    with pytest.raises(ValueError, match=field):
        _chunk(rows=2, **overrides)


def test_a_row_deck_with_a_different_row_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="decks.row_deck"):
        GameDataChunk(
            zones={zone: _empty_zone(2) for zone in GameZone},
            won=np.zeros(2, np.bool_),
            on_play=np.zeros(2, np.bool_),
            num_turns=np.zeros(2, np.int32),
            keys=_keys(2),
            rank=np.full(2, "", object),
            decks=ChunkDecks(decks=(_deck(),), row_deck=np.zeros(3, np.intp)),
        )
