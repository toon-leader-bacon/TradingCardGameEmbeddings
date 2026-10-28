"""CardShelf: loads each game's CardBinder, each DeckBox, each game's
level-2 snapshots and each deck split index at most once, so dojos built
for one run share them."""

from pathlib import Path
from typing import Sequence

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.multi_game_card_lookup import MultiGameCardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.snapshots.card_snapshot import CardSnapshot
from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.data_refinement.snapshots.stamps import DeckSnapshotStamp
from src.dojos.file_managers.deck_split_index import DeckSplitIndex
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

# Deck split indices (and caches derived from a deck split) live here
DECK_SPLIT_CACHE_DIRECTORY = Path("data/splits/contrastive")


class CardShelf:
    """Loads each game's CardBinder, each DeckBox, each game's snapshots
    and each deck split index at most once."""

    def __init__(
        self, deck_split_cache_directory: Path = DECK_SPLIT_CACHE_DIRECTORY
    ) -> None:
        """Inputs: deck_split_cache_directory (where deck_split_index
        caches its files; tests pass a temporary one).
        Side effects: none (nothing loads until asked)."""
        self._deck_split_cache_directory = deck_split_cache_directory
        self._card_binders: dict[GameId, CardBinder] = {}
        self._deck_boxes: dict[Path, DeckBox] = {}
        self._card_snapshots: dict[GameId, CardSnapshot] = {}
        self._deck_snapshots: dict[tuple[GameId, Path], DeckSnapshot] = {}
        self._deck_split_indices: dict[
            tuple[DeckSnapshotStamp, str], DeckSplitIndex
        ] = {}

    @property
    def deck_split_cache_directory(self) -> Path:
        """Where deck split indices, and caches derived from a deck split
        (e.g. staple document frequencies), are kept.

        Inputs: none. Output: Path. Side effects: none. Exceptions: none.
        """
        return self._deck_split_cache_directory

    def card_binder(self, game: GameId) -> CardBinder:
        """game's binder from CardBinder.default_output_path, loaded on
        first request.

        Inputs: game (GameId). Output: CardBinder.
        Side effects: reads the binder file on first call per game.
        Exceptions: FileNotFoundError if the binder file does not exist.

        Example:
            >>> CardShelf().card_binder(GameId.GWENT)
        """
        if game not in self._card_binders:
            path = CardBinder.default_output_path(game)
            if not path.exists():
                raise FileNotFoundError(
                    f"no {game.value} card binder at {path}; run "
                    "scripts/run_card_binder_ingestion.py first"
                )
            self._card_binders[game] = CardBinder.load([path])
        return self._card_binders[game]

    def card_lookup(self, games: Sequence[GameId]) -> MultiGameCardLookup:
        """One CardLookup over games' binders, sharing the ones already
        loaded (no merged copy of any game's cards).

        Inputs: games (non-empty). Output: MultiGameCardLookup.
        Side effects: loads any game's binder not yet loaded.
        Exceptions: FileNotFoundError as card_binder; ValueError if games
            is empty.

        Example:
            >>> CardShelf().card_lookup([GameId.GWENT, GameId.MTG])
        """
        return MultiGameCardLookup({game: self.card_binder(game) for game in games})

    def deck_box(self, path: Path) -> DeckBox:
        """The DeckBox at path, opened read-only on first request.

        Training never writes a deck box, so a stray write through the
        returned box raises instead of changing the file. Its connection
        may be read from any one thread at a time (see
        DeckBox.open_read_only), so dojos built here can be read by a
        prefetch worker.

        Inputs: path (Path). Output: DeckBox.
        Side effects: opens the deck box on first call per path.
        Exceptions: FileNotFoundError if path does not exist.

        Example:
            >>> CardShelf().deck_box(STS_GG_DECK_BOX_PATH)
        """
        if path not in self._deck_boxes:
            if not path.exists():
                raise FileNotFoundError(
                    f"no deck box at {path}; run scripts/run_metrics.py for "
                    "the metric source that writes it"
                )
            self._deck_boxes[path] = DeckBox.open_read_only(path)
        return self._deck_boxes[path]

    def card_snapshot(self, game: GameId) -> CardSnapshot:
        """game's card snapshot from CardSnapshot.default_directory,
        opened on first request and checked against game's binder.

        Inputs: game. Output: CardSnapshot.
        Side effects: loads game's binder if not loaded; maps the snapshot
            on first call per game.
        Exceptions: FileNotFoundError as card_binder; StaleSnapshotError
            if the snapshot is missing or stale.

        Example:
            >>> CardShelf().card_snapshot(GameId.MTG)
        """
        if game not in self._card_snapshots:
            self._card_snapshots[game] = CardSnapshot.open(
                CardSnapshot.default_directory(game), self.card_binder(game), game
            )
        return self._card_snapshots[game]

    def deck_snapshot(self, game: GameId, deck_box_path: Path) -> DeckSnapshot:
        """game's deck snapshot from DeckSnapshot.default_directory, opened
        on first request and checked against the deck box at deck_box_path
        (opened through deck_box(path)) and game's card snapshot. Checking
        both levels here, at dojo construction, fails a run before
        preflight rather than mid-training.

        Inputs: game; deck_box_path (the published box the snapshot was
            built from). Output: DeckSnapshot.
        Side effects: as card_snapshot and deck_box; on first call per
            (game, deck_box_path) counts the box's decks (about 48 s cold
            on MTG) and maps the snapshot.
        Exceptions: FileNotFoundError if the deck box (or binder) is
            missing; StaleSnapshotError if a snapshot file is missing or
            stale.

        Example:
            >>> CardShelf().deck_snapshot(GameId.MTG, DeckBox.default_output_path(GameId.MTG))
        """
        key = (game, deck_box_path)
        if key not in self._deck_snapshots:
            # Level 1 first (its own clear error), then level 2 against it
            deck_box = self.deck_box(deck_box_path)
            self._deck_snapshots[key] = DeckSnapshot.open(
                DeckSnapshot.default_directory(game), self.card_snapshot(game), deck_box
            )
        return self._deck_snapshots[key]

    def deck_split_index(
        self, snapshot: DeckSnapshot, holdout: HoldoutSpec
    ) -> DeckSplitIndex:
        """snapshot's deck split under holdout, loaded or built on first
        request per (snapshot stamp, deck_split_key) and shared by every
        dojo of the game after that; cached on disk under the shelf's
        deck_split_cache_directory.

        Inputs: snapshot (from deck_snapshot); holdout (the run's).
        Output: DeckSplitIndex.
        Side effects: as DeckSplitIndex.load_or_build, on first call per key.
        Exceptions: UnknownDeckSplitError if holdout has no deck split;
            OSError on a cache write failure.

        Example:
            >>> shelf.deck_split_index(decks, holdout)
        """
        key = (snapshot.stamp, holdout.deck_split_key(snapshot.source_game))
        if key not in self._deck_split_indices:
            self._deck_split_indices[key] = DeckSplitIndex.load_or_build(
                snapshot, holdout, self._deck_split_cache_directory
            )
        return self._deck_split_indices[key]
