"""Tests for code_tallies.py's CodeTallies."""

from uuid import uuid4

import numpy as np
import pytest

from src.data_refinement.metrics.seventeenlands.replay_data.code_tallies import (
    CodeTallies,
)

_UUIDS = tuple(uuid4() for _ in range(4))


def _codes(*codes: int) -> np.ndarray:
    return np.array(codes, np.int32)


def test_add_counts_occurrences_and_sums_weights() -> None:
    tallies = CodeTallies("test", 2)

    tallies.add(0, _codes(1, 3, 1))
    tallies.add(1, _codes(1, 3, 1), np.array([2, 5, 4], np.int16))

    uuids, counts = tallies.count_columns(_UUIDS, ("n", "s"), lambda t: t[0] > 0)
    assert uuids == [str(_UUIDS[1]), str(_UUIDS[3])]
    assert counts["n"].tolist() == [2.0, 1.0]
    assert counts["s"].tolist() == [6.0, 5.0]


def test_the_code_axis_grows_across_adds() -> None:
    tallies = CodeTallies("test", 1)

    tallies.add(0, _codes(0))
    tallies.add(0, _codes(2, 0))

    _, counts = tallies.count_columns(_UUIDS, ("n",), lambda t: True)
    assert counts["n"].tolist() == [2.0, 0.0, 1.0]


def test_an_empty_add_changes_nothing() -> None:
    tallies = CodeTallies("test", 1)

    tallies.add(0, _codes())

    uuids, counts = tallies.count_columns((), ("n",), lambda t: True)
    assert uuids == []
    assert counts["n"].size == 0


@pytest.mark.parametrize("tally", [-1, 2])
def test_an_unknown_tally_is_rejected(tally: int) -> None:
    with pytest.raises(ValueError, match="no tally"):
        CodeTallies("test", 2).add(tally, _codes(0))


def test_a_negative_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        CodeTallies("test", 1).add(0, _codes(-1))


def test_weights_must_match_codes() -> None:
    with pytest.raises(ValueError, match="weights"):
        CodeTallies("test", 1).add(0, _codes(0, 1), np.array([1.0]))


def test_tally_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tally_count"):
        CodeTallies("test", 0)


def test_count_columns_needs_one_name_per_tally() -> None:
    with pytest.raises(ValueError, match="names"):
        CodeTallies("test", 2).count_columns(_UUIDS, ("n",), lambda t: True)


def test_count_columns_needs_a_uuid_per_code() -> None:
    tallies = CodeTallies("test", 1)
    tallies.add(0, _codes(3))

    with pytest.raises(ValueError, match="card uuids"):
        tallies.count_columns(_UUIDS[:2], ("n",), lambda t: True)
