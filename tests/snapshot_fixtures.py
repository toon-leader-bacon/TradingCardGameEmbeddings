"""Test helpers that build real level-2 snapshots (and dealers over them)
from small in-memory binders and deck boxes, in a temporary directory."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.snapshots.build import (
    build_card_snapshot,
    build_deck_snapshot,
)
from src.data_refinement.snapshots.card_snapshot import CardSnapshot
from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.dojos.file_managers.deck_box_dealer import DeckBoxDealer
from src.dojos.file_managers.deck_split_index import DeckSplitIndex
from src.schema.card import GenericCard, GenericDeck, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec


def card(game: GameId = GameId.MTG, name: str = "card") -> GenericCard:
    """A fresh card with a random uuid."""
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=game,
        name=name,
        raw_content={"name": name},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def binder_with_cards(count: int, game: GameId = GameId.MTG) -> CardBinder:
    """A binder holding count distinct cards of game."""
    binder = CardBinder()
    for position in range(count):
        binder.create(card(game, f"card {position}"))
    return binder


def stamped_deck_box(
    decks: list[GenericDeck],
    binder: CardBinder,
    directory: Path,
    game: GameId = GameId.MTG,
) -> DeckBox:
    """decks saved to directory/<game>.db stamped with binder's version,
    reopened read-only (as training opens it)."""
    box = DeckBox()
    for deck in decks:
        box.create(deck)
    path = directory / f"{game.value}.db"
    box.save(path, game, binder.version_for(game))
    return DeckBox.open_read_only(path)


def built_snapshots(
    binder: CardBinder, deck_box: DeckBox, root: Path, game: GameId = GameId.MTG
) -> DeckSnapshot:
    """Build game's card and deck snapshots under root and open them."""
    build_card_snapshot(binder, game, root / "cards")
    card_snapshot = CardSnapshot.open(root / "cards", binder, game)
    build_deck_snapshot(deck_box, card_snapshot, root / "decks")
    return DeckSnapshot.open(root / "decks", card_snapshot, deck_box)


def dealer_over(
    decks: list[GenericDeck],
    holdout: HoldoutSpec,
    binder: CardBinder | None = None,
    root: Path | None = None,
    shuffle_seed: int = 0,
) -> DeckBoxDealer:
    """A DeckBoxDealer over decks, through real snapshots and a split index
    built in root (a fresh temporary directory when None)."""
    binder = binder if binder is not None else CardBinder()
    root = root if root is not None else Path(tempfile.mkdtemp())
    box = stamped_deck_box(decks, binder, root)
    snapshot = built_snapshots(binder, box, root)
    split_index = DeckSplitIndex.load_or_build(snapshot, holdout, root / "splits")
    return DeckBoxDealer(snapshot, split_index, shuffle_seed)
