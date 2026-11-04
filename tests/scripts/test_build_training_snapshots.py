"""Tests for scripts/build_training_snapshots.py: builds, skips and rebuilds."""

import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.snapshots.card_snapshot import CardSnapshot
from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.data_refinement.snapshots.stamps import StaleDeckBoxError
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from tests.snapshot_fixtures import binder_with_cards, card, stamped_deck_box

_SCRIPT = Path("scripts/build_training_snapshots.py")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_training_snapshots_script", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def level_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CardBinder:
    """An MTG binder file and stamped deck box under tmp_path, with every
    default path pointed there. Returns the binder."""
    binder = binder_with_cards(2)
    binder.save(tmp_path / "mtg.jsonl", GameId.MTG)
    decks = [
        GenericDeck(uuid4(), GameId.MTG, "deck", list(binder.all_uuids(GameId.MTG)))
    ]
    stamped_deck_box(decks, binder, tmp_path)

    monkeypatch.setattr(
        CardBinder, "default_output_path", lambda game: tmp_path / f"{game.value}.jsonl"
    )
    monkeypatch.setattr(
        DeckBox, "default_output_path", lambda game: tmp_path / f"{game.value}.db"
    )
    monkeypatch.setattr(
        CardSnapshot, "default_directory", lambda game: tmp_path / "cards" / game.value
    )
    monkeypatch.setattr(
        DeckSnapshot, "default_directory", lambda game: tmp_path / "decks" / game.value
    )
    return binder


def test_builds_both_snapshots(
    script: ModuleType, level_one: CardBinder, tmp_path: Path
) -> None:
    script.build_game(GameId.MTG, force=False)

    assert (tmp_path / "cards" / "mtg" / "cards.arrow").exists()
    assert (tmp_path / "decks" / "mtg" / "decks.arrow").exists()


def test_a_second_run_skips_current_snapshots(
    script: ModuleType, level_one: CardBinder, caplog: pytest.LogCaptureFixture
) -> None:
    script.build_game(GameId.MTG, force=False)
    with caplog.at_level(logging.INFO):
        script.build_game(GameId.MTG, force=False)

    messages = [record.getMessage() for record in caplog.records]
    assert "mtg: card snapshot is current" in messages
    assert "mtg: deck snapshot is current" in messages


def test_force_rebuilds(
    script: ModuleType, level_one: CardBinder, caplog: pytest.LogCaptureFixture
) -> None:
    script.build_game(GameId.MTG, force=False)
    with caplog.at_level(logging.INFO):
        script.build_game(GameId.MTG, force=True)

    assert any("wrote 1 decks" in record.getMessage() for record in caplog.records)


def test_a_stale_card_snapshot_is_rebuilt_in_place(
    script: ModuleType,
    level_one: CardBinder,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The stale file was mapped by the failed open; the rebuild replaces it
    script.build_game(GameId.MTG, force=False)
    level_one.create(card())
    level_one.save(tmp_path / "mtg.jsonl", GameId.MTG)

    with caplog.at_level(logging.INFO):
        with pytest.raises(StaleDeckBoxError):
            script.build_game(GameId.MTG, force=False)

    rebuilt = CardSnapshot.open(tmp_path / "cards" / "mtg", level_one, GameId.MTG)
    assert rebuilt.card_count == 4
    assert any("rebuilding the card snapshot" in r.message for r in caplog.records)


def test_a_game_without_a_deck_box_gets_only_cards(
    script: ModuleType,
    level_one: CardBinder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DeckBox, "default_output_path", lambda game: tmp_path / "missing.db"
    )

    script.build_game(GameId.MTG, force=False)

    assert (tmp_path / "cards" / "mtg" / "cards.arrow").exists()
    assert not (tmp_path / "decks").exists()


def test_a_missing_binder_raises(script: ModuleType, level_one: CardBinder) -> None:
    with pytest.raises(FileNotFoundError, match="run_card_binder_ingestion"):
        script.build_game(GameId.GWENT, force=False)


def test_build_all_covers_games_with_binders(
    script: ModuleType, level_one: CardBinder
) -> None:
    assert script._games_with_binders() == [GameId.MTG]
