"""Teacher and load_or_build_teacher: the one entry point the student
side (and scripts/build_teachers.py) calls.

Files live under <settings.directory>/<game>/:
    <stamp tag>.teacher.arrow                 the TeacherTable
    <stamp tag>.<rule tag>.neighbours.arrow   its TeacherNeighbours

A table is rebuilt when no file records the wanted stamp; neighbours are
recomputed when no file records the table's stamp and the rule (cheap
next to counting).
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path


from src.teacher.cooccurrence import count_cooccurrence
from src.teacher.neighbour_rules import NeighbourRule
from src.teacher.teacher_inputs import TeacherInputs
from src.teacher.teacher_method import teacher_method_for
from src.teacher.teacher_neighbours import TeacherNeighbours, find_neighbours
from src.teacher.teacher_settings import TeacherSettings
from src.teacher.teacher_table import (
    StaleTeacherError,
    TeacherStamp,
    TeacherTable,
    stamp_tag_of,
)

_logger = logging.getLogger(__name__)

_TABLE_SUFFIX = ".teacher.arrow"
_NEIGHBOURS_SUFFIX = ".neighbours.arrow"


@dataclass(frozen=True, eq=False)
class Teacher:
    """One game's teacher, as the student side consumes it.

    table: the vectors. neighbours: each row's neighbours; carries
    table.stamp (checked at construction).
    """

    table: TeacherTable
    neighbours: TeacherNeighbours

    def __post_init__(self) -> None:
        """Raise ValueError unless the neighbours index this table."""
        # Validate inputs: the neighbours index this table's rows
        if self.neighbours.teacher_stamp != self.table.stamp:
            raise ValueError("Teacher: the neighbours were found for another table")
        if self.neighbours.row_count != len(self.table.card_uuids):
            raise ValueError("Teacher: the neighbours cover another row count")


def load_or_build_teacher(inputs: TeacherInputs, settings: TeacherSettings) -> Teacher:
    """inputs' game's teacher under settings: read from its cache files
    when their stamps match, otherwise counted, fitted, ranked and cached.

    Inputs:
        inputs: the game's decks with the holdout applied.
        settings: the run's teacher settings.
    Output: Teacher.
    Side effects: may read inputs' snapshot (counting); may write the
        table and neighbours files; logs what it reused or built, and how
        long building took.
    Exceptions: ValueError if inputs' game is not in settings.games or is
        held out, or as count_cooccurrence and the method raise; OSError
        on a write failure.

    Example:
        >>> teacher = load_or_build_teacher(inputs, config.teacher)
        >>> teacher.table.dim
        128
    """
    # Validate inputs: a configured game with decks of its own
    game = inputs.decks.source_game
    _require_buildable(inputs, settings)
    directory = settings.directory / game.value
    stamp = teacher_stamp_for(inputs, settings)

    # The table: cached, else built and cached
    table_path = directory / f"{stamp.tag()}{_TABLE_SUFFIX}"
    table = _cached_table(table_path, stamp)
    if table is None:
        table = _built_table(inputs, settings, stamp)
        table.save(table_path)

    # Its neighbours: cached, else found and cached
    neighbours_path = _neighbours_path(directory, stamp, settings.neighbours)
    neighbours = _cached_neighbours(
        neighbours_path, stamp, settings.neighbours, len(table.card_uuids)
    )
    if neighbours is None:
        neighbours = find_neighbours(table, settings.neighbours)
        neighbours.save(neighbours_path)

    result = Teacher(table, neighbours)
    return result


def teacher_stamp_for(inputs: TeacherInputs, settings: TeacherSettings) -> TeacherStamp:
    """The stamp a teacher of inputs under settings carries.

    Inputs: inputs, settings. Output: TeacherStamp.
    Side effects: none. Exceptions: none.

    Example:
        >>> teacher_stamp_for(inputs, settings).deck_split_key == inputs.deck_split_key
        True
    """
    return TeacherStamp(
        deck_snapshot=inputs.decks.stamp,
        deck_split_key=inputs.deck_split_key,
        card_tier_key=inputs.holdout.card_tier_key(),
        fit_digest=settings.fit_digest(),
    )


def _require_buildable(inputs: TeacherInputs, settings: TeacherSettings) -> None:
    """Raise ValueError if inputs' game is not in settings.games or is in
    inputs.holdout.held_out_games (no TRAIN decks: nothing to count).

    Private helper - single caller is load_or_build_teacher.
    Inputs: inputs (TeacherInputs), settings (TeacherSettings).
    Output: None. Side effects: none.
    Exceptions: ValueError naming the game.
    """
    game = inputs.decks.source_game
    if game not in settings.games:
        raise ValueError(f"{game.value} is not one of teacher.games {settings.games}")
    if game in inputs.holdout.held_out_games:
        raise ValueError(f"{game.value} is held out: it has no TRAIN decks to count")


def _built_table(
    inputs: TeacherInputs, settings: TeacherSettings, stamp: TeacherStamp
) -> TeacherTable:
    """Count, fit and flag confidence: a fresh TeacherTable.

    Private helper - single caller is load_or_build_teacher.
    Inputs: inputs (TeacherInputs), settings (TeacherSettings), stamp
        (TeacherStamp, this table's).
    Output: TeacherTable (confident = deck_counts >= settings.min_deck_count).
    Side effects: reads the snapshot; logs timings.
    Exceptions: as count_cooccurrence and the method's fit.
    """
    started = time.monotonic()
    counts = count_cooccurrence(
        inputs, settings.deck_splits, settings.card_tiers, settings.cooccurrence
    )
    vectors = teacher_method_for(settings.method).fit(counts)
    _logger.info(
        "%s teacher: %d cards from %d decks, dim %d, in %.1f s",
        counts.source_game.value,
        len(counts.card_uuids),
        counts.deck_count,
        vectors.shape[1],
        time.monotonic() - started,
    )
    return TeacherTable(
        source_game=counts.source_game,
        card_uuids=counts.card_uuids,
        vectors=vectors,
        deck_counts=counts.deck_counts,
        confident=counts.deck_counts >= settings.min_deck_count,
        stamp=stamp,
    )


def _cached_table(path: Path, stamp: TeacherStamp) -> TeacherTable | None:
    """The table at path if it exists, reads cleanly and records stamp;
    else None (a stale or malformed file is a miss, logged).

    Private helper - single caller is load_or_build_teacher.
    Inputs: path (Path), stamp (TeacherStamp). Output: TeacherTable | None.
    Side effects: may map path. Exceptions: none.
    """
    if not path.exists():
        return None
    try:
        table = TeacherTable.open(path)
    except (StaleTeacherError, ValueError, OSError) as error:
        _logger.warning("ignoring unreadable teacher table %s: %s", path, error)
        return None
    if table.stamp != stamp:
        _logger.info("teacher table %s is for another stamp; rebuilding", path)
        return None
    _logger.info("reusing teacher table %s", path)
    return table


def _neighbours_path(directory: Path, stamp: TeacherStamp, rule: NeighbourRule) -> Path:
    """<directory>/<stamp tag>.<rule tag>.neighbours.arrow.

    Private helper - single caller is load_or_build_teacher.
    Inputs: directory (Path), stamp (TeacherStamp), rule (NeighbourRule).
    Output: Path. Side effects: none.
    Exceptions: none.
    """
    return (
        directory / f"{stamp.tag()}.{stamp_tag_of(rule.payload())}{_NEIGHBOURS_SUFFIX}"
    )


def _cached_neighbours(
    path: Path, stamp: TeacherStamp, rule: NeighbourRule, row_count: int
) -> TeacherNeighbours | None:
    """The neighbours at path if they exist, read cleanly, and record
    stamp and rule, and cover row_count rows; else None (logged).

    Private helper - single caller is load_or_build_teacher.
    Inputs: path (Path), stamp (TeacherStamp), rule (NeighbourRule),
        row_count (int, the table's rows).
    Output: TeacherNeighbours | None.
    Side effects: may map path. Exceptions: none.
    """
    if not path.exists():
        return None
    try:
        neighbours = TeacherNeighbours.open(path, rule)
    except (StaleTeacherError, ValueError, OSError) as error:
        _logger.warning("ignoring unreadable teacher neighbours %s: %s", path, error)
        return None
    if neighbours.teacher_stamp != stamp:
        _logger.info("teacher neighbours %s are for another stamp; recomputing", path)
        return None
    if neighbours.row_count != row_count:
        _logger.warning("teacher neighbours %s cover another row count", path)
        return None
    return neighbours
