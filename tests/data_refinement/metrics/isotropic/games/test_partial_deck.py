import logging
from pathlib import Path

import pytest

from src.data_refinement.metrics.isotropic.games.game_log_parser import CardQuantity
from src.data_refinement.metrics.isotropic.games.partial_deck import (
    card_uuid_for_quantity_name,
    distinct_card_uuids,
    expand_card_quantity,
    partial_deck_card_uuids,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    card_quantity,
    card_uuid,
    game_header,
    game_header_player,
    game_log,
    turn,
)

_NAMES = ["Copper", "Estate", "Silver", "Gold", "Witch", "Curse", "Oasis"]


def _header():
    return game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 3), game_header_player("b", 30, 3)),
    )


def test_before_turn_one_is_the_starting_deck(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    log = game_log(_header(), (turn("a", 1, bought=(card_quantity("Silver"),)),))

    deck = partial_deck_card_uuids(binder, log, "a", 1)

    assert sorted(deck) == sorted(
        [card_uuid(binder, "Copper")] * 7 + [card_uuid(binder, "Estate")] * 3
    )


def test_folds_in_only_earlier_turns_of_that_player(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    log = game_log(
        _header(),
        (
            turn("a", 1, bought=(card_quantity("Silver"),)),
            turn("b", 1, bought=(card_quantity("Witch"),)),
            turn("a", 2, bought=(card_quantity("Gold"),)),
        ),
    )

    deck = partial_deck_card_uuids(binder, log, "a", 2)

    assert deck.count(card_uuid(binder, "Silver")) == 1
    assert card_uuid(binder, "Gold") not in deck
    assert card_uuid(binder, "Witch") not in deck


def test_gains_and_trashes_are_applied(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    log = game_log(
        _header(),
        (
            turn(
                "a",
                1,
                bought=(card_quantity("Silver"),),
                gained=(card_quantity("Silver"), card_quantity("Curse")),
                trashed=(card_quantity("Coppers", 2),),
            ),
        ),
    )

    deck = partial_deck_card_uuids(binder, log, "a", 2)

    assert deck.count(card_uuid(binder, "Copper")) == 5
    assert deck.count(card_uuid(binder, "Curse")) == 1
    assert len(deck) == 10 + 2 - 2


def test_trashing_a_card_not_in_the_deck_is_ignored(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    log = game_log(_header(), (turn("a", 1, trashed=(card_quantity("Gold"),)),))

    assert len(partial_deck_card_uuids(binder, log, "a", 2)) == 10


def test_unresolvable_gain_is_excluded(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    log = game_log(_header(), (turn("a", 1, bought=(card_quantity("Nonsense"),)),))

    assert len(partial_deck_card_uuids(binder, log, "a", 2)) == 10


def test_expand_card_quantity_repeats_the_resolved_card(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)

    assert (
        expand_card_quantity(binder, CardQuantity("Coppers", 3))
        == [card_uuid(binder, "Copper")] * 3
    )


def test_expand_card_quantity_unresolved_is_empty(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)

    assert expand_card_quantity(binder, CardQuantity("Nonsense", 2)) == []


def test_card_uuid_for_quantity_name_handles_irregular_plural(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)

    assert card_uuid_for_quantity_name(binder, CardQuantity("Oases", 2)) == card_uuid(
        binder, "Oasis"
    )


def test_distinct_card_uuids_dedupes_in_first_seen_order(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)

    result = distinct_card_uuids(
        binder,
        (
            CardQuantity("Coppers", 3),
            CardQuantity("Silver", 1),
            CardQuantity("Copper", 1),
            CardQuantity("Nonsense", 1),
        ),
    )

    assert result == [card_uuid(binder, "Copper"), card_uuid(binder, "Silver")]


def test_other_players_turns_never_leak_in(tmp_path: Path) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    log = game_log(
        game_header(
            "ab",
            ("Witch",),
            (game_header_player("ab", 40, 1), game_header_player("b", 30, 1)),
        ),
        (
            turn("ab", 1, bought=(card_quantity("Gold"),)),
            turn("b", 1, bought=(card_quantity("Witch"),)),
        ),
    )

    deck = partial_deck_card_uuids(binder, log, "b", 2)

    assert card_uuid(binder, "Witch") in deck
    assert card_uuid(binder, "Gold") not in deck


def test_unmatched_name_is_logged_once_per_binder(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    binder = binder_from_card_names(_NAMES, tmp_path)
    log = game_log(
        _header(),
        (
            turn("a", 1, bought=(card_quantity("Nonsense"),)),
            turn("a", 2),
            turn("a", 3),
        ),
    )

    with caplog.at_level(logging.ERROR):
        partial_deck_card_uuids(binder, log, "a", 2)
        partial_deck_card_uuids(binder, log, "a", 3)

    assert caplog.text.count("Nonsense") == 1
