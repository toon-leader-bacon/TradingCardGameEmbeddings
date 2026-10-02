from pathlib import Path

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.spire_codex_runs.extraction_stage import (
    SpireCodexRunsDeckExtractionStage,
)
from src.data_refinement.deck_box.sts2runs.extraction_stage import (
    Sts2RunsDeckExtractionStage,
)
from src.schema.data_source import DataSource
from tests.data_refinement.deck_box.sts2runs.test_extraction_stage import (
    _player,
    _spire_codex_card_binder,
    _write_runs_gz,
)


def _run(run_hash: str, players: list[dict], abandoned: bool = False) -> dict:
    return {"run_hash": run_hash, "was_abandoned": abandoned, "players": players}


def test_reads_every_page_and_keys_decks_by_run_hash(tmp_path: Path) -> None:
    binder = _spire_codex_card_binder(["STRIKE", "DEFEND"])
    _write_runs_gz(
        tmp_path / "page_00000.jsonl.gz", [_run("aa", [_player(["STRIKE"])])]
    )
    _write_runs_gz(
        tmp_path / "page_00001.jsonl.gz",
        [_run("bb", [_player(["DEFEND"]), _player(["STRIKE"])])],
    )
    box = DeckBox()

    changed = SpireCodexRunsDeckExtractionStage().extract(tmp_path, box, binder)

    assert len(changed) == 3
    deck = box.get_by_uuid(changed[0])
    assert deck is not None and deck.provenance is not None
    assert deck.provenance.data_source == DataSource.SPIRE_CODEX
    assert deck.provenance.source_id == "aa:0"


@pytest.mark.parametrize("keep_abandoned, kept", [(True, 1), (False, 0)])
def test_abandoned_runs_are_kept_unless_turned_off(
    tmp_path: Path, keep_abandoned: bool, kept: int
) -> None:
    binder = _spire_codex_card_binder(["STRIKE"])
    _write_runs_gz(
        tmp_path / "page_00000.jsonl.gz",
        [_run("aa", [_player(["STRIKE"])], abandoned=True)],
    )
    stage = SpireCodexRunsDeckExtractionStage(keep_abandoned=keep_abandoned)
    assert len(stage.extract(tmp_path, DeckBox(), binder)) == kept


def test_deck_ids_differ_from_sts2runs_for_the_same_run_id() -> None:
    codex = SpireCodexRunsDeckExtractionStage().deck_uuid("1", 0)
    assert codex != Sts2RunsDeckExtractionStage().deck_uuid("1", 0)


def test_a_directory_without_pages_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SpireCodexRunsDeckExtractionStage().extract(
            tmp_path, DeckBox(), _spire_codex_card_binder([])
        )
