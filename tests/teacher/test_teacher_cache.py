from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from src.schema.game_id import GameId
from src.teacher import teacher_cache as teacher_cache_module
from src.teacher.neighbour_rules import TopKNeighbours
from src.teacher.teacher_cache import load_or_build_teacher, teacher_stamp_for
from src.teacher.teacher_inputs import TeacherInputs
from tests.teacher.teacher_fixtures import (
    all_train_holdout,
    binder_and_cards,
    settings,
    teacher_inputs,
)


def _two_cluster_inputs(tmp_path: Path) -> tuple[TeacherInputs, list[UUID]]:
    binder, cards = binder_and_cards(6)
    first, second = cards[:3], cards[3:]
    decks = [first] * 5 + [second] * 5 + [[cards[2], cards[3]]]
    return teacher_inputs(tmp_path / "decks", binder, decks), cards


def _fail(*args: object, **kwargs: object) -> None:
    raise AssertionError("should have been read from the cache")


class TestLoadOrBuildTeacher:
    def test_builds_a_table_and_its_neighbours(self, tmp_path: Path) -> None:
        inputs, cards = _two_cluster_inputs(tmp_path)

        teacher = load_or_build_teacher(inputs, settings(tmp_path / "teachers"))

        assert set(teacher.table.card_uuids) == set(cards)
        assert teacher.table.dim == 4
        assert teacher.table.stamp == teacher_stamp_for(
            inputs, settings(tmp_path / "teachers")
        )
        assert teacher.neighbours.teacher_stamp == teacher.table.stamp
        # A card's nearest neighbours are its own cluster
        row_of = {card: row for row, card in enumerate(teacher.table.card_uuids)}
        nearest = teacher.neighbours.neighbours_of(row_of[cards[0]]).tolist()
        assert set(nearest) == {row_of[cards[1]], row_of[cards[2]]}
        assert len(list((tmp_path / "teachers" / "mtg").iterdir())) == 2

    def test_second_call_reads_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inputs, _ = _two_cluster_inputs(tmp_path)
        first = load_or_build_teacher(inputs, settings(tmp_path / "teachers"))

        monkeypatch.setattr(teacher_cache_module, "count_cooccurrence", _fail)
        monkeypatch.setattr(teacher_cache_module, "find_neighbours", _fail)
        second = load_or_build_teacher(inputs, settings(tmp_path / "teachers"))

        assert np.array_equal(first.table.vectors, second.table.vectors)
        assert np.array_equal(first.neighbours.indices, second.neighbours.indices)

    def test_new_rule_reuses_the_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inputs, _ = _two_cluster_inputs(tmp_path)
        load_or_build_teacher(inputs, settings(tmp_path / "teachers"))

        monkeypatch.setattr(teacher_cache_module, "count_cooccurrence", _fail)
        teacher = load_or_build_teacher(
            inputs, settings(tmp_path / "teachers", neighbours=TopKNeighbours(1))
        )

        assert all(len(teacher.neighbours.neighbours_of(row)) == 1 for row in range(6))

    def test_low_deck_count_rows_are_not_confident(self, tmp_path: Path) -> None:
        inputs, cards = _two_cluster_inputs(tmp_path)

        teacher = load_or_build_teacher(
            inputs, settings(tmp_path / "teachers", min_deck_count=6)
        )

        # cards 2 and 3 are in 6 decks (their cluster's 5 plus the link)
        confident = {
            card
            for card, flag in zip(teacher.table.card_uuids, teacher.table.confident)
            if flag
        }
        assert confident == {cards[2], cards[3]}

    def test_game_not_configured_raises(self, tmp_path: Path) -> None:
        inputs, _ = _two_cluster_inputs(tmp_path)

        with pytest.raises(ValueError, match="not one of teacher.games"):
            load_or_build_teacher(
                inputs,
                settings(tmp_path / "teachers", games=(GameId.FLESH_AND_BLOOD,)),
            )

    def test_held_out_game_raises(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(2)
        holdout = all_train_holdout(held_out_games=frozenset({GameId.MTG}))
        inputs = teacher_inputs(tmp_path / "decks", binder, [cards], holdout)

        with pytest.raises(ValueError, match="held out"):
            load_or_build_teacher(inputs, settings(tmp_path / "teachers"))
