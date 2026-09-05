from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

from src.data_refinement.card_binder.build import build_or_update_card_binder
from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class _FakeIngestionStage:
    """Test double — creates a fixed set of cards directly on binder,
    matching the new CardIngestionStage contract (owns its own writes,
    no candidate list handed back to a driver)."""

    SOURCE_GAME: ClassVar[GameId] = GameId.MTG

    def __init__(self, rows: list[tuple[str, str, dict]]) -> None:
        # Each row: (name, source_id, raw_content).
        self._rows = rows

    def ingest(self, raw_path: Path, binder: CardBinder) -> list[UUID]:
        changed = []
        for name, source_id, raw_content in self._rows:
            existing = binder.get_by_alias(self.SOURCE_GAME, DataSource.SCRYFALL, source_id)
            if existing is not None:
                continue
            card = GenericCard(
                nocab_uuid=uuid4(),
                source_game=self.SOURCE_GAME,
                name=name,
                raw_content=raw_content,
                provenance=Provenance(
                    data_source=DataSource.SCRYFALL,
                    source_id=source_id,
                    fetched_at=datetime.now(timezone.utc),
                ),
            )
            binder.create(card)
            binder.register_alias(self.SOURCE_GAME, DataSource.SCRYFALL, source_id, card.nocab_uuid)
            changed.append(card.nocab_uuid)
        return changed


class TestBuildOrUpdateCardBinder:
    def test_first_run_creates_binder_from_scratch(self, tmp_path: Path) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        stage = _FakeIngestionStage(
            [("Bolt", "src-1", {"a": 1}), ("Shock", "src-2", {"a": 1})]
        )

        changed = build_or_update_card_binder(Path("unused"), stage, binder_path)

        assert len(changed) == 2
        assert binder_path.exists()
        loaded = CardBinder.load([binder_path])
        assert loaded.get_by_name(GameId.MTG, "Bolt") != []
        assert loaded.get_by_name(GameId.MTG, "Shock") != []

    def test_second_run_adds_to_existing_binder(self, tmp_path: Path) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        first_stage = _FakeIngestionStage([("Bolt", "src-1", {"a": 1})])
        build_or_update_card_binder(Path("unused"), first_stage, binder_path)

        second_stage = _FakeIngestionStage([("Shock", "src-2", {"a": 1})])
        changed = build_or_update_card_binder(Path("unused"), second_stage, binder_path)

        assert len(changed) == 1
        loaded = CardBinder.load([binder_path])
        assert loaded.get_by_name(GameId.MTG, "Bolt") != []
        assert loaded.get_by_name(GameId.MTG, "Shock") != []

    def test_second_run_skips_already_known_alias(self, tmp_path: Path) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        first_stage = _FakeIngestionStage([("Bolt", "src-1", {"a": 1})])
        build_or_update_card_binder(Path("unused"), first_stage, binder_path)
        original = CardBinder.load([binder_path]).get_by_alias(
            GameId.MTG, DataSource.SCRYFALL, "src-1"
        )

        second_stage = _FakeIngestionStage([("Bolt", "src-1", {"a": 1})])
        changed = build_or_update_card_binder(Path("unused"), second_stage, binder_path)

        assert changed == []
        reloaded = CardBinder.load([binder_path]).get_by_alias(
            GameId.MTG, DataSource.SCRYFALL, "src-1"
        )
        assert reloaded.nocab_uuid == original.nocab_uuid

    def test_empty_row_list_produces_empty_binder(self, tmp_path: Path) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        stage = _FakeIngestionStage([])

        changed = build_or_update_card_binder(Path("unused"), stage, binder_path)

        assert changed == []
        assert binder_path.exists()

    def test_uses_ingestion_stage_source_game_not_a_parameter(self, tmp_path: Path) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        stage = _FakeIngestionStage([("Bolt", "src-1", {"a": 1})])

        build_or_update_card_binder(Path("unused"), stage, binder_path)

        loaded = CardBinder.load([binder_path])
        assert list(loaded.all_uuids(GameId.MTG)) != []
        assert list(loaded.all_uuids(GameId.POKEMON)) == []
