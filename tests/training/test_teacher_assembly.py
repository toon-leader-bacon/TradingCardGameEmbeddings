from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.dojos.file_managers.deck_split_index import DeckSplitIndex
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.teacher.teacher_settings import PpmiSvdOptions, TeacherViews
from src.training.teacher_assembly import teacher_for, teachers_for
from tests.snapshot_fixtures import built_snapshots, stamped_deck_box
from tests.teacher.teacher_fixtures import all_train_holdout, binder_and_cards, settings


class _ShelfOver:
    """Stands in for CardShelf: hands out one prepared snapshot and records
    the deck box path it was asked for."""

    def __init__(self, snapshot: DeckSnapshot, split_directory: Path) -> None:
        self.snapshot = snapshot
        self.split_directory = split_directory
        self.asked_paths: list[Path] = []

    def deck_snapshot(self, game: GameId, deck_box_path: Path) -> DeckSnapshot:
        self.asked_paths.append(deck_box_path)
        return self.snapshot

    def deck_split_index(
        self, snapshot: DeckSnapshot, holdout: HoldoutSpec
    ) -> DeckSplitIndex:
        return DeckSplitIndex.load_or_build(snapshot, holdout, self.split_directory)


class TestTeacherFor:
    def test_builds_the_game_teacher_from_the_shelf(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(3)
        decks = [
            GenericDeck(uuid5(NAMESPACE_URL, str(position)), GameId.MTG, "deck", cards)
            for position in range(3)
        ]
        box = stamped_deck_box(decks, binder, tmp_path)
        shelf = _ShelfOver(built_snapshots(binder, box, tmp_path), tmp_path / "splits")

        teacher = teacher_for(
            GameId.MTG,
            shelf,  # type: ignore[arg-type]
            all_train_holdout(),
            settings(tmp_path / "teachers", method=PpmiSvdOptions(dim=2)),
            deck_box_path=tmp_path / "mtg.db",
        )

        assert set(teacher.table.card_uuids) == set(cards)
        assert teacher.table.dim == 2
        assert shelf.asked_paths == [tmp_path / "mtg.db"]


class TestTeachersFor:
    def test_builds_one_teacher_per_view(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(3)
        decks = [
            GenericDeck(uuid5(NAMESPACE_URL, str(position)), GameId.MTG, "deck", cards)
            for position in range(3)
        ]
        box = stamped_deck_box(decks, binder, tmp_path)
        shelf = _ShelfOver(built_snapshots(binder, box, tmp_path), tmp_path / "splits")
        train = settings(tmp_path / "teachers", method=PpmiSvdOptions(dim=2))
        test = replace(train, deck_splits=frozenset({Split.TRAIN, Split.TEST}))

        teachers = teachers_for(
            GameId.MTG,
            shelf,  # type: ignore[arg-type]
            all_train_holdout(),
            TeacherViews({Split.TRAIN: train, Split.TEST: test}),
        )

        assert set(teachers) == {Split.TRAIN, Split.TEST}
        assert (
            teachers[Split.TRAIN].table.stamp.fit_digest
            != teachers[Split.TEST].table.stamp.fit_digest
        )
