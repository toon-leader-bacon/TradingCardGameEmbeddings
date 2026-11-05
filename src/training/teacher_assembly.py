"""A run's TeacherSettings + CardShelf -> one game's Teacher.

The composition root for the teacher: src/teacher never imports the
dojos, so the deck split (DeckSplitIndex, kept with the dojos) is turned
into TeacherInputs here. Called by scripts/build_teachers.py, and by the
teacher-sourced dojo recipes of the student feature.
"""

from pathlib import Path

from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.teacher.teacher_cache import Teacher, load_or_build_teacher
from src.teacher.teacher_inputs import TeacherInputs
from src.teacher.teacher_settings import TeacherSettings
from src.training.dojo_recipes.card_shelf import CardShelf


def teacher_for(
    game: GameId,
    shelf: CardShelf,
    holdout: HoldoutSpec,
    settings: TeacherSettings,
    deck_box_path: Path | None = None,
) -> Teacher:
    """game's teacher under the run's holdout and teacher settings.

    Inputs:
        game: one of settings.games.
        shelf: the run's card shelf (opens the snapshot and split index
            once, shared with the game's dojos).
        holdout: the run's spec.
        settings: the run's teacher settings.
        deck_box_path: the game's published deck box; default
            DeckBox.default_output_path(game).
    Output: Teacher.
    Side effects: as CardShelf.deck_snapshot, CardShelf.deck_split_index
        and load_or_build_teacher (may count decks and write files).
    Exceptions: FileNotFoundError / StaleSnapshotError from the shelf;
        ValueError from load_or_build_teacher.

    Example:
        >>> teacher_for(GameId.MTG, CardShelf(), config.plan.holdout, config.teacher)
    """
    # Open the game's decks and their split under this holdout
    box_path = deck_box_path or DeckBox.default_output_path(game)
    snapshot = shelf.deck_snapshot(game, box_path)
    split_index = shelf.deck_split_index(snapshot, holdout)

    # Hand them to the teacher as plain rows
    inputs = TeacherInputs(
        decks=snapshot,
        deck_rows_by_split={split: split_index.rows_of(split) for split in Split},
        deck_rows_snapshot=split_index.snapshot_stamp,
        deck_split_key=split_index.deck_split_key,
        holdout=holdout,
    )
    return load_or_build_teacher(inputs, settings)
