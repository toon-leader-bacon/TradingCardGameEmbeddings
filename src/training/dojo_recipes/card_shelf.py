"""CardShelf: loads each game's CardBinder and each DeckBox at most once,
so dojos built for one run share them."""

from pathlib import Path
from typing import Sequence

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.multi_game_card_lookup import MultiGameCardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.game_id import GameId


class CardShelf:
    """Loads each game's CardBinder and each DeckBox at most once."""

    def __init__(self) -> None:
        """Side effects: none (nothing loads until asked)."""
        self._card_binders: dict[GameId, CardBinder] = {}
        self._deck_boxes: dict[Path, DeckBox] = {}

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
        """The DeckBox at path, loaded on first request.

        Inputs: path (Path). Output: DeckBox.
        Side effects: reads the deck box on first call per path.
        Exceptions: FileNotFoundError if path does not exist (DeckBox.load
            would silently return an empty box).

        Example:
            >>> CardShelf().deck_box(STS_GG_DECK_BOX_PATH)
        """
        if path not in self._deck_boxes:
            if not path.exists():
                raise FileNotFoundError(
                    f"no deck box at {path}; run scripts/run_metrics.py for "
                    "the metric source that writes it"
                )
            self._deck_boxes[path] = DeckBox.load([path])
        return self._deck_boxes[path]
