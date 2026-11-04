"""CardShelf's level-2 methods: each game's snapshots and each deck split
index are opened once, and both levels are checked."""

from pathlib import Path
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.snapshots.card_snapshot import CardSnapshot
from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.data_refinement.snapshots.stamps import StaleSnapshotError
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.dojo_recipes.card_shelf import CardShelf
from tests.snapshot_fixtures import binder_with_cards, built_snapshots, stamped_deck_box


@pytest.fixture
def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An MTG binder, stamped deck box and snapshots under tmp_path, at the
    paths CardShelf looks in. Returns the deck box path."""
    binder = binder_with_cards(3)
    binder_path = tmp_path / "mtg.jsonl"
    binder.save(binder_path, GameId.MTG)
    decks = [
        GenericDeck(uuid4(), GameId.MTG, "deck", list(binder.all_uuids(GameId.MTG)))
        for _ in range(4)
    ]
    box = stamped_deck_box(decks, binder, tmp_path)
    built_snapshots(binder, box, tmp_path)

    monkeypatch.setattr(CardBinder, "default_output_path", lambda game: binder_path)
    monkeypatch.setattr(
        CardSnapshot, "default_directory", lambda game: tmp_path / "cards"
    )
    monkeypatch.setattr(
        DeckSnapshot, "default_directory", lambda game: tmp_path / "decks"
    )
    return tmp_path / "mtg.db"


class TestDeckSnapshot:
    def test_opens_once_and_shares_the_card_snapshot(self, published: Path) -> None:
        shelf = CardShelf()

        first = shelf.deck_snapshot(GameId.MTG, published)
        second = shelf.deck_snapshot(GameId.MTG, published)

        assert first is second
        assert first.card_snapshot is shelf.card_snapshot(GameId.MTG)
        assert first.deck_count == 4

    def test_a_missing_deck_box_is_file_not_found(self, published: Path) -> None:
        with pytest.raises(FileNotFoundError):
            CardShelf().deck_snapshot(GameId.MTG, published.with_name("missing.db"))

    def test_a_missing_snapshot_is_stale(
        self, published: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            DeckSnapshot, "default_directory", lambda game: published.parent / "none"
        )
        with pytest.raises(StaleSnapshotError, match="build_training_snapshots"):
            CardShelf().deck_snapshot(GameId.MTG, published)


class TestDeckSplitIndex:
    def test_one_per_deck_split_cached_in_the_shelfs_directory(
        self, published: Path
    ) -> None:
        cache_directory = published.parent / "splits"
        shelf = CardShelf(cache_directory)
        snapshot = shelf.deck_snapshot(GameId.MTG, published)
        holdout = HoldoutSpec.no_holdout()

        first = shelf.deck_split_index(snapshot, holdout)
        second = shelf.deck_split_index(snapshot, HoldoutSpec.no_holdout())
        other = shelf.deck_split_index(snapshot, HoldoutSpec.no_holdout(seed=1))

        assert first is second
        assert other is not first
        assert shelf.deck_split_cache_directory == cache_directory
        assert len(list(cache_directory.iterdir())) == 2
