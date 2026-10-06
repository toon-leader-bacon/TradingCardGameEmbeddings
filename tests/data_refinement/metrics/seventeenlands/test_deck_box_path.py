"""Tests for deck_box_path.py's seventeenlands_deck_box_path."""

from pathlib import Path

import pytest

from src.data_refinement.metrics.seventeenlands.deck_box_path import (
    seventeenlands_deck_box_path,
)
from src.data_retrieval.seventeenlands.refs import DataType


def test_each_family_keeps_its_box_beside_its_partitions() -> None:
    assert seventeenlands_deck_box_path(DataType.GAME) == Path(
        "data/metrics/seventeenlands/game_data/deck_box.db"
    )
    assert seventeenlands_deck_box_path(DataType.REPLAY) == Path(
        "data/metrics/seventeenlands/replay_data/deck_box.db"
    )


def test_a_metrics_root_re_roots_the_box(tmp_path: Path) -> None:
    assert seventeenlands_deck_box_path(DataType.GAME, tmp_path) == (
        tmp_path / "seventeenlands" / "game_data" / "deck_box.db"
    )


def test_draft_data_has_no_deck_box() -> None:
    with pytest.raises(ValueError, match="draft_data"):
        seventeenlands_deck_box_path(DataType.DRAFT)
