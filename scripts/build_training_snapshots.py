"""Builds the level-2 training snapshots (src/data_refinement/snapshots/)
for one game, or every game with a card binder: first the card snapshot
from the game's binder, then the deck snapshot from its published deck
box (data/final/decks/<game>.db) and that card snapshot.

Run after card binder and deck box ingestion; training never builds a
snapshot, it refuses a missing or stale one and names this script.

A snapshot whose stamp already matches level 1 is skipped unless
--force. A game with no published deck box gets only a card snapshot.
The MTG deck snapshot reads the whole 40 GB deck box once (minutes).

Usage (from the project root):

    PYTHONPATH=. python3 scripts/build_training_snapshots.py --game mtg
    PYTHONPATH=. python3 scripts/build_training_snapshots.py --all
    PYTHONPATH=. python3 scripts/build_training_snapshots.py --game mtg --force

`--all` builds every game in this process, one after another. One game's
failure is logged and does not stop the rest.
"""

import argparse
import sys
import time

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.snapshots.build import (
    build_card_snapshot,
    build_deck_snapshot,
)
from src.data_refinement.snapshots.card_snapshot import CardSnapshot
from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.data_refinement.snapshots.stamps import StaleSnapshotError
from src.run_logging import (
    add_logging_arguments,
    configure_logging,
    logging_settings_from,
    script_log_path,
    script_logger,
)
from src.schema.game_id import GameId

_logger = script_logger(__file__)


def build_game(game: GameId, force: bool) -> None:
    """Build game's card snapshot, then its deck snapshot if the game has
    a published deck box, skipping each one already current unless force.

    Inputs: game; force (rebuild even when current).
    Output: none.
    Side effects: loads the game's binder and opens its deck box
        read-only; writes data/derived/card_snapshots/<game>/cards.arrow
        and data/derived/deck_snapshots/<game>/decks.arrow; logs.
    Exceptions: FileNotFoundError if the game has no binder;
        StaleDeckBoxError if its deck box was minted against another
        binder; ValueError as the builders.

    Example:
        >>> build_game(GameId.GWENT, force=False)
    """
    # Load the binder (level 1, the source of the card snapshot)
    binder = _load_binder(game)

    # Card snapshot first: the deck snapshot's indices point into it
    card_snapshot = _current_or_rebuilt_card_snapshot(binder, game, force)

    # Deck snapshot, when the game has a published deck box
    box_path = DeckBox.default_output_path(game)
    if not box_path.exists():
        _logger.info("%s: no deck box at %s; card snapshot only", game.value, box_path)
        return
    _build_deck_snapshot_unless_current(
        DeckBox.open_read_only(box_path), card_snapshot, force
    )


def build_all(force: bool) -> None:
    """build_game for every game with a binder file, isolating one game's
    failure (logged with its traceback) from the rest.

    Inputs: force. Output: none.
    Side effects: as build_game per game; exits 1 if any game failed.
    Exceptions: none (SystemExit on failure).
    """
    failed: list[str] = []
    for game in _games_with_binders():
        try:
            build_game(game, force)
        except Exception:
            _logger.exception("%s failed", game.value)
            failed.append(game.value)

    if failed:
        _logger.error("Failed: %s", ", ".join(failed))
        sys.exit(1)


def _load_binder(game: GameId) -> CardBinder:
    """game's binder from CardBinder.default_output_path.
    Inputs: game. Output: CardBinder. Side effects: reads the file.
    Exceptions: FileNotFoundError naming the ingestion script."""
    path = CardBinder.default_output_path(game)
    if not path.exists():
        raise FileNotFoundError(
            f"no {game.value} card binder at {path}; run "
            "scripts/run_card_binder_ingestion.py first"
        )
    return CardBinder.load([path])


def _current_or_rebuilt_card_snapshot(
    binder: CardBinder, game: GameId, force: bool
) -> CardSnapshot:
    """game's card snapshot, rebuilt first when missing, stale or forced.
    Inputs: binder, game, force. Output: CardSnapshot (open).
    Side effects: may write cards.arrow; logs. Exceptions: as
    build_card_snapshot."""
    directory = CardSnapshot.default_directory(game)
    if not force:
        try:
            result = CardSnapshot.open(directory, binder, game)
            _logger.info("%s: card snapshot is current", game.value)
            return result
        except StaleSnapshotError as error:
            _logger.info("%s: rebuilding the card snapshot (%s)", game.value, error)

    stamp = build_card_snapshot(binder, game, directory)
    result = CardSnapshot.open(directory, binder, game)
    _logger.info(
        "%s: wrote %d cards (binder %s...)",
        game.value,
        result.card_count,
        stamp.card_binder_version[:12],
    )
    return result


def _build_deck_snapshot_unless_current(
    deck_box: DeckBox, card_snapshot: CardSnapshot, force: bool
) -> None:
    """Rebuild card_snapshot's game's deck snapshot when missing, stale or
    forced. Inputs: deck_box, card_snapshot, force. Output: none.
    Side effects: may write decks.arrow; logs. Exceptions: as
    build_deck_snapshot."""
    game = card_snapshot.source_game
    directory = DeckSnapshot.default_directory(game)
    if not force:
        try:
            DeckSnapshot.open(directory, card_snapshot, deck_box)
            _logger.info("%s: deck snapshot is current", game.value)
            return
        except StaleSnapshotError as error:
            _logger.info("%s: rebuilding the deck snapshot (%s)", game.value, error)

    started = time.perf_counter()
    build_deck_snapshot(deck_box, card_snapshot, directory)
    decks = DeckSnapshot.open(directory, card_snapshot, deck_box)
    _logger.info(
        "%s: wrote %d decks, %d entries in %.0f s",
        game.value,
        decks.deck_count,
        len(decks.card_indices),
        time.perf_counter() - started,
    )


def _games_with_binders() -> list[GameId]:
    """Every GameId whose CardBinder.default_output_path exists, in enum
    order. Inputs: none. Output: list[GameId]. Side effects: stats files.
    Exceptions: none."""
    return [game for game in GameId if CardBinder.default_output_path(game).exists()]


def parse_args() -> argparse.Namespace:
    """This script's command line: --game or --all, plus --force.
    Inputs: sys.argv. Output: argparse.Namespace. Side effects: exits on
    --help or bad arguments. Exceptions: SystemExit."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--game",
        choices=[game.value for game in GameId],
        help="Build just this game's snapshots.",
    )
    target.add_argument(
        "--all", action="store_true", help="Build every game with a card binder."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild snapshots even when their stamps already match.",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Build the asked-for games' snapshots, logging to this script's log.
    Inputs: sys.argv. Output: none. Side effects: as build_game.
    Exceptions: as build_game for --game; SystemExit(1) if --all had a
    failure."""
    args = parse_args()
    configure_logging(logging_settings_from(args, script_log_path(__file__)))

    if args.game:
        build_game(GameId(args.game), args.force)
        return

    build_all(args.force)


if __name__ == "__main__":
    main()
