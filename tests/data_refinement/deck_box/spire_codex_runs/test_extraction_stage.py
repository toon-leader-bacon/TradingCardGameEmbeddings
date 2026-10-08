import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.spire_codex_runs.extraction_stage import (
    SpireCodexRunsDeckExtractionStage,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _spire_codex_card(card_id: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.SLAY_THE_SPIRE_2,
        name=card_id,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SPIRE_CODEX,
            source_id=card_id,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _spire_codex_card_binder(card_ids: list[str]) -> CardBinder:
    binder = CardBinder()
    for card_id in card_ids:
        card = _spire_codex_card(card_id)
        binder.create(card)
        binder.register_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, card_id, card.nocab_uuid
        )
    return binder


def _player(card_ids: list[str]) -> dict:
    return {"deck": [{"id": f"CARD.{card_id}"} for card_id in card_ids]}


def _run(run_hash: str, players: list[dict], abandoned: bool = False) -> dict:
    return {"run_hash": run_hash, "was_abandoned": abandoned, "players": players}


def _write_runs_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as raw_file:
        for row in rows:
            raw_file.write(json.dumps(row) + "\n")


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
    assert deck.source_game == GameId.SLAY_THE_SPIRE_2
    assert deck.provenance.data_source == DataSource.SPIRE_CODEX
    assert deck.provenance.source_id == "aa:0"


def test_a_deck_holds_the_aliased_cards_of_its_player(tmp_path: Path) -> None:
    binder = _spire_codex_card_binder(["STRIKE", "DEFEND"])
    _write_runs_gz(
        tmp_path / "page_00000.jsonl.gz",
        [_run("aa", [_player(["STRIKE", "DEFEND", "STRIKE"])])],
    )
    box = DeckBox()

    (changed,) = SpireCodexRunsDeckExtractionStage().extract(tmp_path, box, binder)

    deck = box.get_by_uuid(changed)
    assert deck is not None
    strike = binder.get_by_alias(
        GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "STRIKE"
    )
    assert strike is not None
    assert deck.card_nocab_uuids.count(strike.nocab_uuid) == 2


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


def test_a_single_page_file_can_be_read_directly(tmp_path: Path) -> None:
    binder = _spire_codex_card_binder(["STRIKE"])
    page = tmp_path / "runs.json.gz"
    _write_runs_gz(page, [_run("aa", [_player(["STRIKE"])])])

    changed = SpireCodexRunsDeckExtractionStage().extract(page, DeckBox(), binder)

    assert len(changed) == 1


def test_a_directory_without_pages_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SpireCodexRunsDeckExtractionStage().extract(
            tmp_path, DeckBox(), _spire_codex_card_binder([])
        )


class TestUnknownCards:
    def test_unresolvable_card_falls_back_to_unknown_sentinel(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _spire_codex_card_binder(["STRIKE"])
        unknown = binder.ensure_unknown_card(GameId.SLAY_THE_SPIRE_2)
        box = DeckBox()
        page = tmp_path / "page_00000.jsonl.gz"
        _write_runs_gz(page, [_run("aa", [_player(["STRIKE", "MISSING_CARD"])])])

        with caplog.at_level(logging.ERROR):
            changed = SpireCodexRunsDeckExtractionStage().extract(page, box, binder)

        deck = box.get_by_uuid(changed[0])
        assert deck is not None
        assert unknown.nocab_uuid in deck.card_nocab_uuids
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_raises_runtime_error_when_unknown_sentinel_not_seeded(
        self, tmp_path: Path
    ) -> None:
        binder = _spire_codex_card_binder(["STRIKE"])
        page = tmp_path / "page_00000.jsonl.gz"
        _write_runs_gz(page, [_run("aa", [_player(["MISSING_CARD"])])])

        with pytest.raises(RuntimeError):
            SpireCodexRunsDeckExtractionStage().extract(page, DeckBox(), binder)


class TestIdempotentReruns:
    def test_reextracting_same_pages_changes_nothing(self, tmp_path: Path) -> None:
        binder = _spire_codex_card_binder(["STRIKE"])
        box = DeckBox()
        page = tmp_path / "page_00000.jsonl.gz"
        _write_runs_gz(page, [_run("aa", [_player(["STRIKE"])])])
        stage = SpireCodexRunsDeckExtractionStage()

        first = stage.extract(page, box, binder)
        second = stage.extract(page, box, binder)

        assert len(first) == 1
        assert second == []
        assert len(list(box.all_decks(GameId.SLAY_THE_SPIRE_2))) == 1

    def test_a_newly_aliased_card_updates_the_deck_not_duplicates_it(
        self, tmp_path: Path
    ) -> None:
        binder = _spire_codex_card_binder(["STRIKE"])
        unknown = binder.ensure_unknown_card(GameId.SLAY_THE_SPIRE_2)
        box = DeckBox()
        page = tmp_path / "page_00000.jsonl.gz"
        _write_runs_gz(page, [_run("aa", [_player(["STRIKE", "DEFEND"])])])
        stage = SpireCodexRunsDeckExtractionStage()
        (deck_uuid,) = stage.extract(page, box, binder)

        # The binder gains the previously unknown card
        defend = _spire_codex_card("DEFEND")
        binder.create(defend)
        binder.register_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, "DEFEND", defend.nocab_uuid
        )
        second = stage.extract(page, box, binder)

        assert second == [deck_uuid]
        assert len(list(box.all_decks(GameId.SLAY_THE_SPIRE_2))) == 1
        deck = box.get_by_uuid(deck_uuid)
        assert deck is not None
        assert defend.nocab_uuid in deck.card_nocab_uuids
        assert unknown.nocab_uuid not in deck.card_nocab_uuids

    def test_a_reordered_card_list_is_not_reported_as_changed(
        self, tmp_path: Path
    ) -> None:
        binder = _spire_codex_card_binder(["STRIKE", "DEFEND"])
        box = DeckBox()
        stage = SpireCodexRunsDeckExtractionStage()
        first_page = tmp_path / "first.jsonl.gz"
        _write_runs_gz(first_page, [_run("aa", [_player(["STRIKE", "DEFEND"])])])
        assert len(stage.extract(first_page, box, binder)) == 1

        # card_nocab_uuids is an unordered multiset
        reordered = tmp_path / "reordered.jsonl.gz"
        _write_runs_gz(reordered, [_run("aa", [_player(["DEFEND", "STRIKE"])])])

        assert stage.extract(reordered, box, binder) == []

    def test_deck_ids_are_stable_and_distinct_per_player(self) -> None:
        stage = SpireCodexRunsDeckExtractionStage()
        assert stage.deck_uuid("aa", 0) == stage.deck_uuid("aa", 0)
        assert stage.deck_uuid("aa", 0) != stage.deck_uuid("aa", 1)
        assert stage.deck_uuid("aa", 0) != stage.deck_uuid("bb", 0)
