from pathlib import Path

import pytest

from src.data_refinement.metrics.isotropic.games.row_utils import (
    is_single_pile_ending,
    pile_card_uuid_for_name,
    winner_player,
)
from tests.data_refinement.metrics.isotropic.games._helpers import (
    binder_from_card_names,
    card_uuid,
    game_header,
    game_header_player,
)


def test_winner_player_finds_matching_nick() -> None:
    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 20), game_header_player("b", 30, 20)),
    )

    assert winner_player(header).nick == "a"


def test_winner_player_raises_on_internal_inconsistency() -> None:
    header = game_header("nobody", ("Witch",), (game_header_player("a", 40, 20),))

    with pytest.raises(ValueError):
        winner_player(header)


def test_is_single_pile_ending_true_for_one_pile() -> None:
    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Provinces",),
    )
    assert is_single_pile_ending(header) is True


def test_is_single_pile_ending_false_for_multiple_piles() -> None:
    header = game_header(
        "a",
        ("Witch",),
        (game_header_player("a", 40, 20),),
        exhausted_pile_names=("Duke", "Monument", "Duchy"),
    )
    assert is_single_pile_ending(header) is False


def test_pile_card_uuid_for_name_plain_plural(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Province"], tmp_path)
    assert pile_card_uuid_for_name(binder, "Provinces") == card_uuid(binder, "Province")


def test_pile_card_uuid_for_name_ies_plural(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Duchy"], tmp_path)
    assert pile_card_uuid_for_name(binder, "Duchies") == card_uuid(binder, "Duchy")


def test_pile_card_uuid_for_name_ves_plural(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Wharf"], tmp_path)
    assert pile_card_uuid_for_name(binder, "Wharves") == card_uuid(binder, "Wharf")


def test_pile_card_uuid_for_name_unchanged_plural(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Ironworks"], tmp_path)
    assert pile_card_uuid_for_name(binder, "Ironworks") == card_uuid(
        binder, "Ironworks"
    )


def test_pile_card_uuid_for_name_es_and_s_both_apply(tmp_path: Path) -> None:
    # "Duchesses" ends in both "es" and "s" - the "es"-stripped
    # candidate ("Duchess") must win over the "s"-stripped one
    # ("Duchesse", not a real card).
    binder = binder_from_card_names(["Duchess"], tmp_path)
    assert pile_card_uuid_for_name(binder, "Duchesses") == card_uuid(binder, "Duchess")


def test_pile_card_uuid_for_name_es_plural(tmp_path: Path) -> None:
    # "Witch" pluralizes with "+es", not a plain "+s" - confirmed live
    # ("Witches" appears as a real exhausted-pile name).
    binder = binder_from_card_names(["Witch"], tmp_path)
    assert pile_card_uuid_for_name(binder, "Witches") == card_uuid(binder, "Witch")


def test_pile_card_uuid_for_name_unresolved_returns_none(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Witch"], tmp_path)
    assert pile_card_uuid_for_name(binder, "Not A Real Card") is None


def test_pile_card_uuid_for_name_is_plural(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Oasis"], tmp_path)
    assert pile_card_uuid_for_name(binder, "Oases") == card_uuid(binder, "Oasis")


def test_pile_card_uuid_for_name_pluralized_head_before_of(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Horn of Plenty"], tmp_path)
    assert pile_card_uuid_for_name(binder, "Horns of Plenty") == card_uuid(
        binder, "Horn of Plenty"
    )
