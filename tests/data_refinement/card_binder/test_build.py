from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.build import IngestSummary, build_or_update_card_binder
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.ingestion import ExternalIdentifier, IngestedCandidate
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


class _FakeIngestionStage:
    """Test double — returns a fixed candidate list, no real parsing."""

    def __init__(self, candidates: list[IngestedCandidate]) -> None:
        self._candidates = candidates

    def ingest(self, raw_path: Path, source_game: GameId) -> list[IngestedCandidate]:
        return self._candidates


def _candidate(
    name: str,
    source_id: str,
    raw_content: dict,
    aliases: list[ExternalIdentifier] | None = None,
) -> IngestedCandidate:
    card = GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=source_id,
            fetched_at=datetime.now(timezone.utc),
        ),
    )
    return IngestedCandidate(card=card, aliases=aliases or [])


class TestBuildOrUpdateCardBinder:
    def test_first_run_creates_binder_from_scratch(self, tmp_path: Path) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        stage = _FakeIngestionStage(
            [_candidate("Bolt", "src-1", {"a": 1}), _candidate("Shock", "src-2", {"a": 1})]
        )

        summary = build_or_update_card_binder(Path("unused"), GameId.MTG, stage, binder_path)

        assert summary == IngestSummary(inserted=2, content_updated=0, kept_existing=0)
        assert binder_path.exists()
        loaded = CardBinder.load([binder_path])
        assert loaded.get_by_name(GameId.MTG, "Bolt") is not None
        assert loaded.get_by_name(GameId.MTG, "Shock") is not None

    def test_second_run_updates_existing_binder(self, tmp_path: Path) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        first_stage = _FakeIngestionStage([_candidate("Bolt", "src-1", {"a": 1})])
        build_or_update_card_binder(Path("unused"), GameId.MTG, first_stage, binder_path)

        second_stage = _FakeIngestionStage(
            [
                _candidate("Bolt", "src-2", {"a": 1, "b": 2}),
                _candidate("Shock", "src-3", {"a": 1}),
            ]
        )
        summary = build_or_update_card_binder(
            Path("unused"), GameId.MTG, second_stage, binder_path
        )

        assert summary == IngestSummary(inserted=1, content_updated=1, kept_existing=0)
        loaded = CardBinder.load([binder_path])
        bolt = loaded.get_by_name(GameId.MTG, "Bolt")
        assert bolt.raw_content == {"a": 1, "b": 2}
        assert (
            loaded.get_by_alias(GameId.MTG, DataSource.SCRYFALL, "src-1").nocab_uuid
            == bolt.nocab_uuid
        )

    def test_tallies_kept_existing(self, tmp_path: Path) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        first_stage = _FakeIngestionStage([_candidate("Bolt", "src-1", {"a": 1, "b": 2})])
        build_or_update_card_binder(Path("unused"), GameId.MTG, first_stage, binder_path)

        poorer_stage = _FakeIngestionStage([_candidate("Bolt", "src-2", {"a": 1})])
        summary = build_or_update_card_binder(
            Path("unused"), GameId.MTG, poorer_stage, binder_path
        )

        assert summary == IngestSummary(inserted=0, content_updated=0, kept_existing=1)

    def test_empty_candidate_list_produces_empty_binder(self, tmp_path: Path) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        stage = _FakeIngestionStage([])

        summary = build_or_update_card_binder(Path("unused"), GameId.MTG, stage, binder_path)

        assert summary == IngestSummary(inserted=0, content_updated=0, kept_existing=0)
        assert binder_path.exists()

    def test_extra_aliases_are_registered_against_the_stored_card(
        self, tmp_path: Path
    ) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        stage = _FakeIngestionStage(
            [
                _candidate(
                    "Bolt",
                    "src-1",
                    {"a": 1},
                    aliases=[ExternalIdentifier(DataSource.ARENA, "76497")],
                )
            ]
        )

        build_or_update_card_binder(Path("unused"), GameId.MTG, stage, binder_path)

        loaded = CardBinder.load([binder_path])
        bolt = loaded.get_by_name(GameId.MTG, "Bolt")
        assert loaded.get_by_alias(GameId.MTG, DataSource.ARENA, "76497") == bolt

    def test_extra_alias_registers_against_survivor_on_collision(
        self, tmp_path: Path
    ) -> None:
        binder_path = tmp_path / "mtg.jsonl"
        first_stage = _FakeIngestionStage([_candidate("Bolt", "src-1", {"a": 1, "b": 2})])
        build_or_update_card_binder(Path("unused"), GameId.MTG, first_stage, binder_path)

        second_stage = _FakeIngestionStage(
            [
                _candidate(
                    "Bolt",
                    "src-2",
                    {"a": 1},  # poorer — loses richness comparison
                    aliases=[ExternalIdentifier(DataSource.ARENA, "76497")],
                )
            ]
        )
        build_or_update_card_binder(Path("unused"), GameId.MTG, second_stage, binder_path)

        loaded = CardBinder.load([binder_path])
        survivor = loaded.get_by_name(GameId.MTG, "Bolt")
        assert survivor.raw_content == {"a": 1, "b": 2}
        assert loaded.get_by_alias(GameId.MTG, DataSource.ARENA, "76497") == survivor
