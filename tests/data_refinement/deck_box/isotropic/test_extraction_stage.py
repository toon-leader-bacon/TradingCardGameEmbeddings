import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.isotropic.extraction_stage import (
    IsotropicDeckExtractionStage,
)
from src.schema.data_source import DataSource
from src.schema.game_id import GameId
from tests.data_refinement.metrics.isotropic.summary._helpers import (
    binder_from_card_names,
    card_uuid,
    player_entry,
    summary_row,
)


def _write_archive(path: Path, rows: list[dict]) -> None:
    content = "\n".join(json.dumps(row) for row in rows).encode("utf-8")
    with tarfile.open(path, "w:bz2") as archive:
        info = tarfile.TarInfo(name="games-20200101.json")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))


def test_stores_each_finished_players_full_deck(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Village", "Smithy"], tmp_path)
    players = [
        player_entry("a", 1, {"Village": 2, "Smithy": 1}),
        player_entry("b", 2, {"Smithy": 3}),
    ]
    _write_archive(tmp_path / "x-summary.tar.bz2", [summary_row([], players)])
    box = DeckBox()

    created = IsotropicDeckExtractionStage().extract(tmp_path, box, binder)

    assert len(created) == 2
    decks = [box.get_by_uuid(uuid) for uuid in created]
    assert Counter(decks[0].card_nocab_uuids) == {  # type: ignore[union-attr]
        card_uuid(binder, "Village"): 2,
        card_uuid(binder, "Smithy"): 1,
    }
    assert decks[0].source_game == GameId.DOMINION  # type: ignore[union-attr]
    assert decks[0].provenance.data_source == DataSource.ISOTROPIC  # type: ignore


def test_skips_resigned_players_and_partially_known_decks(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Village"], tmp_path)
    players = [
        player_entry("resigned", 2, resigned=True),  # no end block
        player_entry("unknown", 1, {"Village": 1, "Not A Card": 1}),
    ]
    archive = tmp_path / "x-summary.tar.bz2"
    _write_archive(archive, [summary_row([], players)])

    box = DeckBox()
    assert IsotropicDeckExtractionStage().extract(archive, box, binder) == []
    assert list(box.all_uuids(GameId.DOMINION)) == []


def test_identical_decks_and_reruns_are_stored_once(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Village"], tmp_path)
    same = [player_entry("a", 1, {"Village": 1}), player_entry("b", 2, {"Village": 1})]
    _write_archive(tmp_path / "x-summary.tar.bz2", [summary_row([], same)])
    box = DeckBox()
    stage = IsotropicDeckExtractionStage()

    assert len(stage.extract(tmp_path, box, binder)) == 1
    assert stage.extract(tmp_path, box, binder) == []


def test_a_directory_without_archives_raises(tmp_path: Path) -> None:
    binder = binder_from_card_names(["Village"], tmp_path)
    with pytest.raises(FileNotFoundError):
        IsotropicDeckExtractionStage().extract(tmp_path, DeckBox(), binder)
