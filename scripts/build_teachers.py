"""Builds (or confirms cached) every teacher a run config asks for, before
the run: the config's `teacher:` section under its `holdout:`, one game
after another (src/teacher/).

Optional: a training run builds a missing teacher itself the first time
a teacher-sourced dojo is constructed. This script pre-warms the cache
(MTG counts ~10M decks) and is the way to inspect a teacher build alone.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/build_teachers.py --config <run config yaml>
    PYTHONPATH=. python3 scripts/build_teachers.py --config <yaml> --game mtg

One game's failure is logged and does not stop the rest.
"""

import argparse
import sys
from pathlib import Path

from src.run_logging import (
    add_logging_arguments,
    configure_logging,
    logging_settings_from,
    script_log_path,
    script_logger,
)
from src.schema.game_id import GameId
from src.teacher.teacher_settings import TeacherSettings
from src.training.dojo_catalog import DOJO_CATALOG
from src.training.dojo_recipes.card_shelf import CardShelf
from src.training.run_config import RunConfig, parse_run_config, read_config_document
from src.training.teacher_assembly import teacher_for

_logger = script_logger(__file__)


def build_teachers(config: RunConfig, games: tuple[GameId, ...]) -> list[GameId]:
    """Build or reuse each game's teacher under config's holdout and
    teacher settings, isolating one game's failure from the rest.

    Inputs: config (must have a teacher section); games (a subset of
        config.teacher.games).
    Output: list[GameId], the games that failed (empty on success).
    Side effects: as teacher_for per game (may count decks and write
        data/teachers/<game>/); logs each game's outcome.
    Exceptions: ValueError if config has no teacher section.

    Example:
        >>> build_teachers(config, config.teacher.games)
        []
    """
    settings = _require_teacher_settings(config)
    shelf = CardShelf()
    result: list[GameId] = []

    # One game at a time; one failure does not stop the rest
    for game in games:
        try:
            teacher = teacher_for(game, shelf, config.plan.holdout, settings)
            _logger.info(
                "%s: %d rows (%d confident), dim %d",
                game.value,
                len(teacher.table.card_uuids),
                int(teacher.table.confident.sum()),
                teacher.table.dim,
            )
        except Exception:
            _logger.exception("%s failed", game.value)
            result.append(game)
    return result


def _require_teacher_settings(config: RunConfig) -> TeacherSettings:
    """config.teacher, or ValueError naming the missing section.

    Private helper - single caller is build_teachers.
    Inputs: config. Output: TeacherSettings. Side effects: none.
    Exceptions: ValueError.
    """
    if config.teacher is None:
        raise ValueError("the run config has no teacher: section")
    return config.teacher


def _chosen_games(
    settings: TeacherSettings, game_text: str | None
) -> tuple[GameId, ...]:
    """settings.games, or just --game's (which must be one of them).

    Private helper - single caller is main.
    Inputs: settings, game_text (None for all). Output: tuple[GameId, ...].
    Side effects: none. Exceptions: ValueError for a game not configured.
    """
    if game_text is None:
        return settings.games
    game = GameId(game_text)
    if game not in settings.games:
        raise ValueError(f"--game {game.value} is not one of teacher.games")
    return (game,)


def parse_args() -> argparse.Namespace:
    """The command line: --config, optional --game, logging flags.

    Inputs: sys.argv. Output: argparse.Namespace. Side effects: none.
    Exceptions: SystemExit on bad arguments.

    Example:
        >>> parse_args().config  # argv: --config run.yaml
        PosixPath('run.yaml')
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="A run config YAML.")
    parser.add_argument("--game", help="Build only this game's teacher.")
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Build the config's teachers, logging to this script's log.

    Inputs: sys.argv. Output: none. Side effects: as build_teachers.
    Exceptions: as parse_run_config; SystemExit(1) if any game failed.
    """
    args = parse_args()
    configure_logging(logging_settings_from(args, script_log_path(__file__)))

    # Parse the whole run config: the teacher shares the run's holdout
    config = parse_run_config(read_config_document(args.config), tuple(DOJO_CATALOG))
    games = _chosen_games(_require_teacher_settings(config), args.game)
    failed = build_teachers(config, games)
    if failed:
        _logger.error("Failed: %s", ", ".join(game.value for game in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
