from datetime import datetime, timezone
from uuid import uuid4

from src.data_refinement.card_binder.merge_strategies import (
    keep_existing,
    keep_incoming,
    keep_incoming_if_content_differs,
    keep_longer_content,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(raw_content: dict) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="Bolt",
        raw_content=raw_content,
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id="src-1",
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestKeepExisting:
    def test_returns_existing_unchanged(self) -> None:
        existing = _card({"a": 1})
        candidate = _card({"a": 1, "b": 2})

        result = keep_existing(existing, candidate)

        assert result == existing

    def test_preserves_existing_uuid(self) -> None:
        existing = _card({"a": 1})
        candidate = _card({"a": 1, "b": 2})

        result = keep_existing(existing, candidate)

        assert result.nocab_uuid == existing.nocab_uuid
        assert result.nocab_uuid != candidate.nocab_uuid


class TestKeepIncoming:
    def test_takes_candidate_content(self) -> None:
        existing = _card({"a": 1})
        candidate = _card({"a": 1, "b": 2})

        result = keep_incoming(existing, candidate)

        assert result.raw_content == candidate.raw_content
        assert result.provenance == candidate.provenance

    def test_preserves_existing_uuid_never_candidates(self) -> None:
        existing = _card({"a": 1})
        candidate = _card({"a": 1, "b": 2})

        result = keep_incoming(existing, candidate)

        assert result.nocab_uuid == existing.nocab_uuid
        assert result.nocab_uuid != candidate.nocab_uuid


class TestKeepLongerContent:
    def test_candidate_wins_when_strictly_larger(self) -> None:
        existing = _card({"a": 1})
        candidate = _card({"a": 1, "b": 2})

        result = keep_longer_content(existing, candidate)

        assert result.raw_content == candidate.raw_content
        assert result.nocab_uuid == existing.nocab_uuid

    def test_existing_wins_when_candidate_smaller(self) -> None:
        existing = _card({"a": 1, "b": 2})
        candidate = _card({"a": 1})

        result = keep_longer_content(existing, candidate)

        assert result.raw_content == existing.raw_content
        assert result.nocab_uuid == existing.nocab_uuid

    def test_existing_wins_on_exact_tie(self) -> None:
        existing = _card({"a": 1, "b": 2})
        candidate = _card({"a": 1, "b": 2})

        result = keep_longer_content(existing, candidate)

        assert result.nocab_uuid == existing.nocab_uuid

    def test_never_returns_candidates_own_uuid(self) -> None:
        existing = _card({"a": 1})
        candidate = _card({"a": 1, "b": 2, "c": 3})

        result = keep_longer_content(existing, candidate)

        assert result.nocab_uuid != candidate.nocab_uuid


class TestKeepIncomingIfContentDiffers:
    def test_identical_content_returns_existing_including_its_provenance(self) -> None:
        existing = _card({"a": 1})
        candidate = _card({"a": 1})

        result = keep_incoming_if_content_differs(existing, candidate)

        assert result == existing
        assert result.provenance == existing.provenance

    def test_different_content_takes_incoming_content_and_keeps_existing_uuid(
        self,
    ) -> None:
        existing = _card({"a": 1, "b": 2})
        candidate = _card({"a": 9})

        result = keep_incoming_if_content_differs(existing, candidate)

        assert result.raw_content == {"a": 9}
        assert result.nocab_uuid == existing.nocab_uuid
        assert result.provenance == candidate.provenance

    def test_shorter_incoming_content_still_wins(self) -> None:
        existing = _card({"text": "a long stale text"})
        candidate = _card({"text": "new"})

        assert keep_incoming_if_content_differs(existing, candidate).raw_content == {
            "text": "new"
        }

    def test_different_name_alone_counts_as_a_change(self) -> None:
        existing = _card({"a": 1})
        candidate = _card({"a": 1})
        renamed = GenericCard(
            nocab_uuid=candidate.nocab_uuid,
            source_game=candidate.source_game,
            name="Renamed",
            raw_content=candidate.raw_content,
            provenance=candidate.provenance,
        )

        result = keep_incoming_if_content_differs(existing, renamed)

        assert result.name == "Renamed"
