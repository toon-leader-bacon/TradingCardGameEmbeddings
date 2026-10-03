"""Tests for keyed_card_tallies.py's KeyedCardTallies."""

from uuid import uuid4

import numpy as np
import pytest

from src.data_refinement.metrics.seventeenlands.draft_data.keyed_card_tallies import (
    KeyedCardTallies,
)

_OWLBEAR, _MORNINGSTAR = uuid4(), uuid4()
_NAMES = ("nocab_uuid", "pick_number")
_COUNTS = ("in_pack", "picked")


def _increments(present: list[list[int]], picked: list[list[int]]) -> np.ndarray:
    return np.stack([np.array(present, bool), np.array(picked, bool)])


def test_rows_are_tallied_per_stratum_key() -> None:
    tallies = KeyedCardTallies("test", 2)

    tallies.add(
        (_OWLBEAR, _MORNINGSTAR),
        (np.array([0, 1, 0]),),
        _increments([[1, 1], [1, 0], [1, 0]], [[1, 0], [0, 0], [0, 0]]),
    )

    keys, counts = tallies.count_columns(_NAMES, _COUNTS, lambda t: t[0] > 0)
    rows = {
        (card, pick): (n, p)
        for card, pick, n, p in zip(
            keys["nocab_uuid"], keys["pick_number"], counts["in_pack"], counts["picked"]
        )
    }
    assert rows == {
        (str(_OWLBEAR), 0): (2.0, 1.0),
        (str(_MORNINGSTAR), 0): (1.0, 0.0),
        (str(_OWLBEAR), 1): (1.0, 0.0),
    }


def test_string_and_int_strata_together() -> None:
    tallies = KeyedCardTallies("test", 2)

    tallies.add(
        (_OWLBEAR,),
        (np.array([0, 0]), np.array(["gold", "mythic"], object)),
        _increments([[1], [1]], [[1], [0]]),
    )

    keys, _ = tallies.count_columns(
        ("nocab_uuid", "pick_number", "rank"), _COUNTS, lambda t: True
    )
    assert sorted(keys["rank"]) == ["gold", "mythic"]
    assert all(isinstance(value, int) for value in keys["pick_number"])


def test_no_strata_files_every_row_under_one_key() -> None:
    tallies = KeyedCardTallies("test", 2)

    tallies.add((_OWLBEAR,), (), _increments([[1], [1]], [[1], [0]]))

    keys, counts = tallies.count_columns(("nocab_uuid",), _COUNTS, lambda t: True)
    assert keys == {"nocab_uuid": [str(_OWLBEAR)]}
    assert counts["in_pack"].tolist() == [2.0]


def test_chunks_add_up() -> None:
    tallies = KeyedCardTallies("test", 2)
    for _ in range(3):
        tallies.add((_OWLBEAR,), (np.array([4]),), _increments([[1]], [[1]]))

    _, counts = tallies.count_columns(_NAMES, _COUNTS, lambda t: True)

    assert counts["in_pack"].tolist() == [3.0]


def test_two_columns_naming_one_card_both_count() -> None:
    tallies = KeyedCardTallies("test", 2)

    tallies.add((_OWLBEAR, _OWLBEAR), (np.array([0]),), _increments([[1, 1]], [[0, 0]]))

    _, counts = tallies.count_columns(_NAMES, _COUNTS, lambda t: True)
    assert counts["in_pack"].tolist() == [2.0]


def test_an_empty_chunk_adds_nothing() -> None:
    tallies = KeyedCardTallies("test", 2)

    tallies.add((_OWLBEAR,), (np.array([], np.int64),), np.zeros((2, 0, 1), bool))

    keys, counts = tallies.count_columns(_NAMES, _COUNTS, lambda t: True)
    assert keys == {"nocab_uuid": [], "pick_number": []}
    assert counts["in_pack"].tolist() == []


def test_a_different_layout_is_rejected() -> None:
    tallies = KeyedCardTallies("test", 2)
    tallies.add((_OWLBEAR,), (np.array([0]),), _increments([[1]], [[0]]))

    with pytest.raises(ValueError, match="differ from the first chunk"):
        tallies.add((_MORNINGSTAR,), (np.array([1]),), _increments([[1]], [[0]]))


def test_a_wrong_shape_is_rejected() -> None:
    tallies = KeyedCardTallies("test", 2)

    with pytest.raises(ValueError, match="increments shape"):
        tallies.add((_OWLBEAR,), (np.array([0]),), np.zeros((1, 1, 1), bool))


def test_tally_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tally_count"):
        KeyedCardTallies("test", 0)
