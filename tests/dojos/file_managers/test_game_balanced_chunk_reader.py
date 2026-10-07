"""Tests for game_balanced_chunk_reader.py's GameBalancedChunkReader."""

import random
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from src.dojos.file_managers.game_balanced_chunk_reader import GameBalancedChunkReader


def _write(path: Path, games: list[str]) -> Path:
    frame = pd.DataFrame(
        {
            "nocab_uuid": [str(index) for index in range(len(games))],
            "source_game": games,
            "label": ["tier_1" if game != "skip" else "other" for game in games],
        }
    )
    frame.to_parquet(path, index=False)
    return path


def _games_drawn(reader: GameBalancedChunkReader) -> list[str]:
    return [game for chunk in reader for game in chunk["source_game"]]


class TestGameBalancedChunkReader:
    def test_a_pass_yields_as_many_rows_as_the_file_holds(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "t.parquet", ["mtg"] * 30 + ["gwent"] * 5)

        reader = GameBalancedChunkReader(path, "source_game", 8, random.Random(0))

        assert len(_games_drawn(reader)) == 35

    def test_each_value_is_drawn_about_equally_often(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "t.parquet", ["mtg"] * 900 + ["gwent"] * 100)

        reader = GameBalancedChunkReader(path, "source_game", 64, random.Random(0))
        counts = Counter(_games_drawn(reader))

        assert abs(counts["mtg"] - counts["gwent"]) < 120

    def test_a_small_value_is_repeated_not_dropped(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "t.parquet", ["mtg"] * 50 + ["gwent"])

        reader = GameBalancedChunkReader(path, "source_game", 10, random.Random(0))
        gwent_rows = [
            row
            for chunk in reader
            for row in chunk["nocab_uuid"][chunk["source_game"] == "gwent"]
        ]

        assert len(gwent_rows) > 1
        assert set(gwent_rows) == {"50"}

    def test_chunks_are_at_most_chunk_rows_long(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "t.parquet", ["mtg"] * 20 + ["gwent"] * 3)

        reader = GameBalancedChunkReader(path, "source_game", 7, random.Random(0))

        assert [len(chunk) for chunk in reader] == [7, 7, 7, 2]

    def test_the_same_seed_gives_the_same_pass(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "t.parquet", ["mtg"] * 20 + ["gwent"] * 3)

        first = _games_drawn(
            GameBalancedChunkReader(path, "source_game", 5, random.Random(1))
        )
        second = _games_drawn(
            GameBalancedChunkReader(path, "source_game", 5, random.Random(1))
        )

        assert first == second

    def test_successive_passes_differ(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "t.parquet", ["mtg"] * 40 + ["gwent"] * 40)
        reader = GameBalancedChunkReader(path, "source_game", 5, random.Random(1))

        assert _games_drawn(reader) != _games_drawn(reader)

    def test_a_missing_balance_column_raises_key_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "t.parquet", ["mtg"])
        reader = GameBalancedChunkReader(path, "no_such_column", 5, random.Random(0))

        with pytest.raises(KeyError):
            list(reader)

    def test_keep_row_removes_rows_before_balancing(self, tmp_path: Path) -> None:
        games = ["mtg"] * 50 + ["gwent"] * 10 + ["skip"] * 100
        path = _write(tmp_path / "t.parquet", games)

        reader = GameBalancedChunkReader(
            path,
            "source_game",
            16,
            random.Random(0),
            keep_row=lambda frame: frame["label"] != "other",
        )
        counts = Counter(_games_drawn(reader))

        assert "skip" not in counts
        assert sum(counts.values()) == 60
