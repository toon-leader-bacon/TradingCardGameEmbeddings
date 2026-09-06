import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.sts_gg.extraction_stage import (
    StsGgDeckExtractionStage,
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


def _run_row(run_id: str, card_ids: list[str]) -> dict:
    return {
        "id": run_id,
        "win": True,
        "deck": [{"id": f"CARD.{card_id}"} for card_id in card_ids],
    }


def _write_runs_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as raw_file:
        for row in rows:
            raw_file.write(json.dumps(row) + "\n")


class TestExtract:
    def test_all_resolvable_cards_creates_deck(self, tmp_path: Path) -> None:
        binder = _spire_codex_card_binder(["STRIKE", "DEFEND"])
        box = DeckBox()
        raw_path = tmp_path / "runs.jsonl"
        _write_runs_jsonl(raw_path, [_run_row("run-1", ["STRIKE", "DEFEND"])])
        stage = StsGgDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        assert len(changed_uuids) == 1
        deck = box.get_by_uuid(changed_uuids[0])
        assert deck is not None
        assert deck.source_game == GameId.SLAY_THE_SPIRE_2
        expected_uuids = {
            binder.get_by_alias(
                GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, card_id
            ).nocab_uuid
            for card_id in ["STRIKE", "DEFEND"]
        }
        assert set(deck.card_nocab_uuids) == expected_uuids

    def test_unresolvable_card_falls_back_to_unknown_sentinel(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _spire_codex_card_binder(["STRIKE"])
        unknown = binder.ensure_unknown_card(GameId.SLAY_THE_SPIRE_2)
        box = DeckBox()
        raw_path = tmp_path / "runs.jsonl"
        _write_runs_jsonl(raw_path, [_run_row("run-1", ["STRIKE", "MISSING_CARD"])])
        stage = StsGgDeckExtractionStage()

        with caplog.at_level(logging.ERROR):
            changed_uuids = stage.extract(raw_path, box, binder)

        deck = box.get_by_uuid(changed_uuids[0])
        assert unknown.nocab_uuid in deck.card_nocab_uuids
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_reextracting_same_raw_path_is_idempotent(self, tmp_path: Path) -> None:
        binder = _spire_codex_card_binder(["STRIKE"])
        box = DeckBox()
        raw_path = tmp_path / "runs.jsonl"
        _write_runs_jsonl(raw_path, [_run_row("run-1", ["STRIKE"])])
        stage = StsGgDeckExtractionStage()

        first = stage.extract(raw_path, box, binder)
        second = stage.extract(raw_path, box, binder)

        assert len(first) == 1
        assert second == []
        assert len(list(box.all_decks(GameId.SLAY_THE_SPIRE_2))) == 1

    def test_reextracting_after_resolution_changes_updates_not_duplicates(
        self, tmp_path: Path
    ) -> None:
        binder = _spire_codex_card_binder(["STRIKE"])
        unknown = binder.ensure_unknown_card(GameId.SLAY_THE_SPIRE_2)
        box = DeckBox()
        raw_path = tmp_path / "runs.jsonl"
        _write_runs_jsonl(raw_path, [_run_row("run-1", ["STRIKE", "DEFEND"])])
        stage = StsGgDeckExtractionStage()

        first = stage.extract(raw_path, box, binder)
        assert len(first) == 1
        deck_uuid = first[0]
        deck_after_first = box.get_by_uuid(deck_uuid)
        assert unknown.nocab_uuid in deck_after_first.card_nocab_uuids

        # spire_codex's CardBinder gains the previously-unresolvable card.
        defend_card = _spire_codex_card("DEFEND")
        binder.create(defend_card)
        binder.register_alias(
            GameId.SLAY_THE_SPIRE_2,
            DataSource.SPIRE_CODEX,
            "DEFEND",
            defend_card.nocab_uuid,
        )

        second = stage.extract(raw_path, box, binder)

        assert second == [deck_uuid]
        assert len(list(box.all_decks(GameId.SLAY_THE_SPIRE_2))) == 1
        deck_after_second = box.get_by_uuid(deck_uuid)
        assert defend_card.nocab_uuid in deck_after_second.card_nocab_uuids
        assert unknown.nocab_uuid not in deck_after_second.card_nocab_uuids

    def test_reordered_but_same_content_card_list_is_not_reported_as_changed(
        self, tmp_path: Path
    ) -> None:
        binder = _spire_codex_card_binder(["STRIKE", "DEFEND"])
        box = DeckBox()
        first_path = tmp_path / "runs.jsonl"
        _write_runs_jsonl(first_path, [_run_row("run-1", ["STRIKE", "DEFEND"])])
        stage = StsGgDeckExtractionStage()
        first = stage.extract(first_path, box, binder)
        assert len(first) == 1

        # Same run, same cards, different raw order — must not be
        # reported as a change (card_nocab_uuids is an unordered
        # multiset, see src/schema/card.py).
        reordered_path = tmp_path / "runs_reordered.jsonl"
        _write_runs_jsonl(reordered_path, [_run_row("run-1", ["DEFEND", "STRIKE"])])

        second = stage.extract(reordered_path, box, binder)

        assert second == []
        assert len(list(box.all_decks(GameId.SLAY_THE_SPIRE_2))) == 1

    def test_raises_runtime_error_when_unknown_sentinel_not_seeded(
        self, tmp_path: Path
    ) -> None:
        binder = _spire_codex_card_binder(["STRIKE"])
        box = DeckBox()
        raw_path = tmp_path / "runs.jsonl"
        _write_runs_jsonl(raw_path, [_run_row("run-1", ["MISSING_CARD"])])
        stage = StsGgDeckExtractionStage()

        with pytest.raises(RuntimeError):
            stage.extract(raw_path, box, binder)
