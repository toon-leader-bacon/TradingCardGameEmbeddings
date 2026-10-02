"""Tests for card_column_tallies.py's CardColumnTallies."""

from uuid import uuid4

import numpy as np
import pytest

from src.data_refinement.metrics.seventeenlands.game_data.card_column_tallies import (
    CardColumnTallies,
)


def test_columns_naming_one_card_are_summed_per_card() -> None:
    owlbear, morningstar = uuid4(), uuid4()
    tallies = CardColumnTallies("test", 2, np.int64)

    tallies.add((owlbear, morningstar, owlbear), np.array([[1, 2, 3], [4, 5, 6]]))

    per_card = tallies.per_card()
    assert list(per_card) == [owlbear, morningstar]
    assert per_card[owlbear].tolist() == [4, 10]
    assert per_card[morningstar].tolist() == [2, 5]


def test_chunks_add_up() -> None:
    owlbear = uuid4()
    tallies = CardColumnTallies("test", 1, np.float64)

    tallies.add((owlbear,), np.array([[1.5]]))
    tallies.add((owlbear,), np.array([[2.0]]))

    assert tallies.per_card()[owlbear].tolist() == [3.5]


def test_no_chunks_means_no_cards() -> None:
    assert CardColumnTallies("test", 1, np.int64).per_card() == {}


def test_a_different_layout_is_rejected() -> None:
    tallies = CardColumnTallies("test", 1, np.int64)
    tallies.add((uuid4(),), np.array([[1]]))

    with pytest.raises(ValueError, match="differ from the first chunk"):
        tallies.add((uuid4(),), np.array([[1]]))


def test_a_wrong_increment_shape_is_rejected() -> None:
    tallies = CardColumnTallies("test", 2, np.int64)

    with pytest.raises(ValueError, match="increments shape"):
        tallies.add((uuid4(),), np.array([[1]]))


def test_per_card_does_not_alias_the_running_tallies() -> None:
    owlbear = uuid4()
    tallies = CardColumnTallies("test", 1, np.int64)
    tallies.add((owlbear,), np.array([[1]]))

    tallies.per_card()[owlbear][0] = 99

    assert tallies.per_card()[owlbear].tolist() == [1]
