import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.play_gwent.extraction_stage import (
    PlayGwentDeckExtractionStage,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _gwent_one_card(card_id: int) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.GWENT,
        name=f"card-{card_id}",
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.GWENT_ONE,
            source_id=str(card_id),
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _gwent_one_card_binder(card_ids: list[int]) -> CardBinder:
    binder = CardBinder()
    for card_id in card_ids:
        card = _gwent_one_card(card_id)
        binder.create(card)
        binder.register_alias(
            GameId.GWENT, DataSource.GWENT_ONE, str(card_id), card.nocab_uuid
        )
    return binder


def _guide_row(
    guide_id: int, card_template_ids: list[int], name: str | None = None
) -> dict:
    row: dict = {
        "id": guide_id,
        "deck": {"id": guide_id * 10, "srcCardTemplates": card_template_ids},
    }
    if name is not None:
        row["name"] = name
    return row


def _write_guides_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as raw_file:
        for row in rows:
            raw_file.write(json.dumps(row) + "\n")


class TestExtract:
    def test_all_resolvable_cards_creates_deck(self, tmp_path: Path) -> None:
        binder = _gwent_one_card_binder([202338, 202376])
        box = DeckBox()
        raw_path = tmp_path / "guides.jsonl"
        _write_guides_jsonl(
            raw_path, [_guide_row(407697, [202338, 202376, 202376], "My Deck")]
        )
        stage = PlayGwentDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        assert len(changed_uuids) == 1
        deck = box.get_by_uuid(changed_uuids[0])
        assert deck is not None
        assert deck.source_game == GameId.GWENT
        assert deck.name == "My Deck"
        card_338_uuid = binder.get_by_alias(
            GameId.GWENT, DataSource.GWENT_ONE, "202338"
        ).nocab_uuid
        card_376_uuid = binder.get_by_alias(
            GameId.GWENT, DataSource.GWENT_ONE, "202376"
        ).nocab_uuid
        assert deck.card_nocab_uuids == [card_338_uuid, card_376_uuid, card_376_uuid]

    def test_missing_or_empty_name_falls_back_to_synthetic_name(
        self, tmp_path: Path
    ) -> None:
        binder = _gwent_one_card_binder([202338])
        box = DeckBox()
        raw_path = tmp_path / "guides.jsonl"
        _write_guides_jsonl(
            raw_path,
            [
                _guide_row(1, [202338], name=None),
                _guide_row(2, [202338], name=""),
            ],
        )
        stage = PlayGwentDeckExtractionStage()

        changed_uuids = stage.extract(raw_path, box, binder)

        names = {box.get_by_uuid(uuid).name for uuid in changed_uuids}
        assert names == {
            "playgwent.com guide 1",
            "playgwent.com guide 2",
        }

    def test_unresolvable_card_falls_back_to_unknown_sentinel(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = _gwent_one_card_binder([202338])
        unknown = binder.ensure_unknown_card(GameId.GWENT)
        box = DeckBox()
        raw_path = tmp_path / "guides.jsonl"
        _write_guides_jsonl(raw_path, [_guide_row(407697, [202338, 999999])])
        stage = PlayGwentDeckExtractionStage()

        with caplog.at_level(logging.ERROR):
            changed_uuids = stage.extract(raw_path, box, binder)

        deck = box.get_by_uuid(changed_uuids[0])
        assert unknown.nocab_uuid in deck.card_nocab_uuids
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_reextracting_same_raw_path_is_idempotent(self, tmp_path: Path) -> None:
        binder = _gwent_one_card_binder([202338])
        box = DeckBox()
        raw_path = tmp_path / "guides.jsonl"
        _write_guides_jsonl(raw_path, [_guide_row(407697, [202338])])
        stage = PlayGwentDeckExtractionStage()

        first = stage.extract(raw_path, box, binder)
        second = stage.extract(raw_path, box, binder)

        assert len(first) == 1
        assert second == []
        assert len(list(box.all_decks(GameId.GWENT))) == 1

    def test_reextracting_after_resolution_changes_updates_not_duplicates(
        self, tmp_path: Path
    ) -> None:
        binder = _gwent_one_card_binder([202338])
        unknown = binder.ensure_unknown_card(GameId.GWENT)
        box = DeckBox()
        raw_path = tmp_path / "guides.jsonl"
        _write_guides_jsonl(raw_path, [_guide_row(407697, [202338, 202376])])
        stage = PlayGwentDeckExtractionStage()

        first = stage.extract(raw_path, box, binder)
        assert len(first) == 1
        deck_uuid = first[0]
        deck_after_first = box.get_by_uuid(deck_uuid)
        assert unknown.nocab_uuid in deck_after_first.card_nocab_uuids

        # gwent.one's CardBinder gains the previously-unresolvable card.
        new_card = _gwent_one_card(202376)
        binder.create(new_card)
        binder.register_alias(
            GameId.GWENT, DataSource.GWENT_ONE, "202376", new_card.nocab_uuid
        )

        second = stage.extract(raw_path, box, binder)

        assert second == [deck_uuid]
        assert len(list(box.all_decks(GameId.GWENT))) == 1
        deck_after_second = box.get_by_uuid(deck_uuid)
        assert new_card.nocab_uuid in deck_after_second.card_nocab_uuids
        assert unknown.nocab_uuid not in deck_after_second.card_nocab_uuids

    def test_reordered_but_same_content_card_list_is_not_reported_as_changed(
        self, tmp_path: Path
    ) -> None:
        binder = _gwent_one_card_binder([202338, 202376])
        box = DeckBox()
        first_path = tmp_path / "guides.jsonl"
        _write_guides_jsonl(first_path, [_guide_row(407697, [202338, 202376])])
        stage = PlayGwentDeckExtractionStage()
        first = stage.extract(first_path, box, binder)
        assert len(first) == 1

        # Same guide, same cards, different raw order — must not be
        # reported as a change (card_nocab_uuids is an unordered
        # multiset, see src/schema/card.py).
        reordered_path = tmp_path / "guides_reordered.jsonl"
        _write_guides_jsonl(reordered_path, [_guide_row(407697, [202376, 202338])])

        second = stage.extract(reordered_path, box, binder)

        assert second == []
        assert len(list(box.all_decks(GameId.GWENT))) == 1

    def test_renamed_guide_does_not_update_stored_deck_name(
        self, tmp_path: Path
    ) -> None:
        binder = _gwent_one_card_binder([202338, 202376])
        box = DeckBox()
        first_path = tmp_path / "guides.jsonl"
        _write_guides_jsonl(
            first_path, [_guide_row(407697, [202338], name="Original Name")]
        )
        stage = PlayGwentDeckExtractionStage()
        first = stage.extract(first_path, box, binder)
        deck_uuid = first[0]

        # Same guide id, content changed (a new card) AND the guide's
        # own title changed — the stored deck's name must stay as
        # first seen (see _extract_guide()'s docstring: name is never
        # refreshed on an update()).
        renamed_path = tmp_path / "guides_renamed.jsonl"
        _write_guides_jsonl(
            renamed_path,
            [_guide_row(407697, [202338, 202376], name="Renamed Deck")],
        )
        second = stage.extract(renamed_path, box, binder)

        assert second == [deck_uuid]
        deck = box.get_by_uuid(deck_uuid)
        assert deck.name == "Original Name"

    def test_raises_runtime_error_when_unknown_sentinel_not_seeded(
        self, tmp_path: Path
    ) -> None:
        binder = _gwent_one_card_binder([202338])
        box = DeckBox()
        raw_path = tmp_path / "guides.jsonl"
        _write_guides_jsonl(raw_path, [_guide_row(407697, [999999])])
        stage = PlayGwentDeckExtractionStage()

        with pytest.raises(RuntimeError):
            stage.extract(raw_path, box, binder)
