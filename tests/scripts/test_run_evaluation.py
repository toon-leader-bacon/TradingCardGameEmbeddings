import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

_SCRIPT = Path("scripts/run_evaluation.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_evaluation_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _holdout(deck_split_ratios: tuple[float, float, float] | None) -> HoldoutSpec:
    return HoldoutSpec(
        seed=0,
        tier_ratios=(1, 0, 0),
        held_out_games=frozenset(),
        deck_split_ratios=deck_split_ratios,
    )


def _write_gwent_box(path: Path, deck_count: int) -> None:
    box = DeckBox.load([path])
    for _ in range(deck_count):
        cards = [uuid4() for _ in range(5)]
        box.create(GenericDeck(uuid4(), GameId.GWENT, "deck", cards))
    box.flush()


@pytest.fixture
def box_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "gwent.db"
    monkeypatch.setattr(DeckBox, "default_output_path", staticmethod(lambda game: path))
    return path


def test_build_deck_auc_samples_validation_decks(
    box_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    script = _load_script()
    _write_gwent_box(box_path, deck_count=6)
    with caplog.at_level(logging.INFO):
        analysis = script.build_deck_auc(
            _holdout((0, 0, 1)), script.budget_for(smoke=True)
        )
    assert analysis is not None
    assert analysis.name == "deck_retrieval_auc"
    # Every deck is VALIDATION and has five distinct cards
    assert "6 VALIDATION decks" in caplog.text


def test_build_deck_auc_skips_without_a_deck_box(
    box_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    script = _load_script()
    with caplog.at_level(logging.WARNING):
        result = script.build_deck_auc(
            _holdout((0, 0, 1)), script.budget_for(smoke=True)
        )
    assert result is None
    assert not box_path.exists()
    assert "no deck box" in caplog.text


def test_build_deck_auc_skips_an_unknown_deck_split(
    box_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    script = _load_script()
    _write_gwent_box(box_path, deck_count=3)
    with caplog.at_level(logging.WARNING):
        result = script.build_deck_auc(_holdout(None), script.budget_for(smoke=True))
    assert result is None
    assert "no deck split" in caplog.text


def test_build_deck_auc_skips_when_no_deck_qualifies(box_path: Path) -> None:
    script = _load_script()
    _write_gwent_box(box_path, deck_count=3)
    assert (
        script.build_deck_auc(_holdout((1, 0, 0)), script.budget_for(smoke=True))
        is None
    )
